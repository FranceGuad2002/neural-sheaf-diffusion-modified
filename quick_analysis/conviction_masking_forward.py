#!/usr/bin/env python3
"""
Conviction forward (diagnostic) experiment (Alessio's protocol) — manifest-driven runner.

Which experiments run is read from quick_analysis/conviction_manifest.txt (one
line per dataset/model/hyperparameter combo), filtered by --model/--normalised;
nothing is discovered by globbing the results/ tree.

Train the SNN normally, then — without retraining — apply one of three
interventions to a subset of nodes and measure accuracy / confidence drop on
the *full test set*:
  masking  — zero (or mean-replace) the selected nodes' input features
  pruning  — remove the selected nodes + incident edges, forward-pass on the
             smaller graph (frozen weights reloaded into a freshly-shaped model,
             since edge_index/graph_size are baked into the model at construction)
  gating   — scale the selected nodes' OUTGOING messages via --gate_mode:
             hard — zero them (budget sweep, like masking/pruning)
             soft — continuously attenuate every train node's message by its
                    own conviction (no budget sweep, one point per fold)

Eight selection strategies at each k% budget for masking/pruning/hard-gating.
Candidates are drawn from TRAIN nodes only
(val/test are never touched), and k = frac * |pool|, so the test set used for
evaluation is identical across all strategies and budgets:
  high             — top-k% by conviction               (deterministic)
  low              — bottom-k% by conviction             (deterministic)
  deg_high         — top-k% by degree                    (deterministic)
  deg_low          — bottom-k% by degree                 (deterministic)
  deg_matched_high — k random nodes matching high's degree dist (1 draw/fold)
  deg_matched_low  — k random nodes matching low's degree dist  (1 draw/fold)
  conf_low         — bottom-k% by prediction confidence  (deterministic)
  random           — k uniformly random nodes            (1 draw/fold)

Conviction score (--conviction_score):
  last   — X_L norm, the last diffusion layer only (default)
  avg    — mean raw norm over post-diffusion layers 1..L (layer 0, the
           pre-diffusion encoder output, is excluded since it hasn't been
           touched by any message-passing step)
  avg_z  — same layer range as avg, but each layer's norms are z-scored
           (zero mean, unit std, across all nodes) before averaging, so no
           single layer's scale can dominate the mean

Output folder:
    quick_analysis/Conviction_experiments/forward_experiment/<intervention>/<conviction_score>/<model>/normalised_{true|false}/
(pruning also writes a diagnostics JSON alongside the plot: nodes/edges
remaining, connected components, class balance per split, split composition
drift — Alessio's required table.)

Usage (from repo root):
    python quick_analysis/conviction_masking_forward.py --model GeneralSheaf --normalised true --intervention masking
    python quick_analysis/conviction_masking_forward.py --model BundleSheaf  --normalised false --intervention pruning --conviction_score avg
"""

import sys
import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from torch_geometric.utils import degree as pyg_degree

from utils.conviction import (
    JOINT_MODELS, TYPE_SHORT, map_tag, weights_path, load_masks,
    load_fold_rankings, select_candidates, draw_rng, model_classes,
    load_manifest_experiments, mask_features, prune_nodes, build_message_gate,
    load_conviction, load_confidence, forward_eval,
    AUC_DATASETS, metric_key, tables_path, table_meta, write_table, mean_se,
)

# ── constants ─────────────────────────────────────────────────────────────────

MASK_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30]

STRATEGIES  = ['high',      'low',       'deg_high',  'deg_low',   'deg_matched_high',   'deg_matched_low',   'conf_high',   'conf_low',  'random']
LABELS      = ['High conv', 'Low conv',  'High deg',  'Low deg',   'Deg-matched (high)', 'Deg-matched (low)', 'High conf',   'Low conf',  'Random']
COLORS      = ['firebrick', 'steelblue', 'darkorange','seagreen',  'goldenrod',          'purple',            'darkcyan',    'teal',      'gray']
LINESTYLES  = ['-',         '--',        '-',         '--',        ':',                  ':',                 '-',           '--',        ':']
MARKERS     = ['o',         'o',         's',         's',         'o',                  'D',                 'P',           'v',         '^']

# ── evaluation ────────────────────────────────────────────────────────────────

def _eval_masking(model, data, mask_idx, mask_mode, device, is_auc=False):
    """Zero/mean-replace mask_idx's features, forward through the SAME model
    (graph unchanged), evaluate on the full test set (masking never touches
    test, so the exclusion below is a defensive no-op, not load-bearing)."""
    masked_data = mask_features(data, mask_idx, mode=mask_mode)
    test_idx = masked_data.test_mask.nonzero(as_tuple=True)[0]
    keep     = ~torch.isin(test_idx, mask_idx.to(device))
    eval_idx = test_idx[keep]
    acc, conf, auc = forward_eval(model, masked_data, eval_idx, device, is_auc)
    return acc, conf, auc, None


def _eval_pruning(model_cls, args, state_dict, data, prune_idx, device, is_auc=False):
    """Remove prune_idx + incident edges, rebuild a model shaped for the
    smaller graph, load the SAME frozen weights into it (no retraining —
    see conversation: graph_size/edge_index are baked in at construction,
    so the original model object can't forward-pass a different graph),
    evaluate on the (relabeled, but never-pruned) test nodes."""
    new_data, diagnostics = prune_nodes(data, prune_idx)
    new_data = new_data.to(device)
    new_args = dict(args)
    new_args['graph_size'] = new_data.num_nodes
    new_model = model_cls(new_data.edge_index, new_args).to(device)
    new_model.load_state_dict(state_dict)
    eval_idx = new_data.test_mask.nonzero(as_tuple=True)[0]
    acc, conf, auc = forward_eval(new_model, new_data, eval_idx, device, is_auc)
    return acc, conf, auc, diagnostics


def _eval_gating_hard(model, data, conviction, gate_idx, device, is_auc=False):
    """Zero the selected nodes' OUTGOING messages only (gate=0.0); their own
    self-term and every other node's messages are untouched (see _apply_gate
    in disc_models.py). Same graph and same model object as the frozen
    original — no reconstruction needed, unlike pruning."""
    n = data.num_nodes
    gate = build_message_gate(n, conviction, gate_idx, mode='hard', device=device)
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    keep     = ~torch.isin(test_idx, gate_idx.to(device))
    eval_idx = test_idx[keep]
    acc, conf, auc = forward_eval(model, data, eval_idx, device, is_auc, gate=gate)
    return acc, conf, auc, None


def _eval_gating_soft(model, data, conviction, train_idx, device, is_auc=False, min_scale=0.1):
    """Continuously attenuate EVERY train node's outgoing message by its own
    conviction — no candidate selection, no budget, the whole train pool is
    gated in one shot (see conversation: this is what distinguishes soft
    gating from hard gating/masking/pruning, which all sweep a selected
    subset at several budgets). No exclusion needed for eval_idx since
    train_idx and test_mask are disjoint by construction."""
    n = data.num_nodes
    gate = build_message_gate(n, conviction, train_idx, mode='soft', min_scale=min_scale, device=device)
    eval_idx = data.test_mask.nonzero(as_tuple=True)[0]
    return forward_eval(model, data, eval_idx, device, is_auc, gate=gate)

# ── per-fold experiment ───────────────────────────────────────────────────────

def _run_fold(args, dataset, model_cls, fold, device, intervention, mask_mode, gate_mode='hard'):
    data   = dataset[0].clone()
    masks  = load_masks(args, fold)
    data.train_mask = masks['train_mask']
    data.val_mask   = masks['val_mask']
    data.test_mask  = masks['test_mask']
    data   = data.to(device)
    n      = data.num_nodes

    wp = weights_path(args, fold)
    if not os.path.exists(wp):
        print(f"  fold {fold}: weights not found — skipping"); return None
    model = model_cls(data.edge_index, args).to(device)
    model.save_other_laplacian = False
    state_dict = torch.load(wp, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    is_auc = args.get('dataset') in AUC_DATASETS
    score  = args.get('conviction_score', 'last')

    # Soft gating has no candidate selection or budget sweep — the entire
    # train pool is gated in one shot — so it takes a completely different,
    # simpler path than masking/pruning/hard-gating's STRATEGIES x MASK_LEVELS
    # loop below, and returns a differently-shaped result.
    if intervention == 'gating' and gate_mode == 'soft':
        train_idx  = data.train_mask.nonzero(as_tuple=True)[0].cpu()
        empty      = torch.tensor([], dtype=torch.long)
        # Three continuous indicators, each gated identically: within the train
        # pool the lowest value maps to min_scale, the highest to ~1.0 (see
        # build_message_gate). Conviction is the original; confidence and degree
        # are the same recipe applied to a different per-node score.
        indicators = {
            'conviction': load_conviction(args, fold, score=score),
            'confidence': load_confidence(args, fold),
            'degree':     pyg_degree(data.edge_index[0].cpu(), num_nodes=n).float(),
        }
        # Baseline is one no-op pass (empty gate ignores the score vector).
        acc0, conf0, auc0 = _eval_gating_soft(model, data, indicators['conviction'], empty, device, is_auc)
        result = {'baseline': {'acc': acc0, 'conf': conf0, 'auc': auc0}}
        msg = f"  fold {fold} done (soft gate)  |  baseline acc={acc0:.3f}"
        for name, scores in indicators.items():
            a, c, u = _eval_gating_soft(model, data, scores, train_idx, device, is_auc)
            result[f'soft_gate_{name}'] = {'acc': a, 'conf': c, 'auc': u}
            msg += f"  {name}={a:.3f}"
        print(msg)
        return result

    rankings     = load_fold_rankings(args, fold, data.edge_index, n, score=score, pool='train')
    pool_idx     = rankings['pool_idx']
    conviction   = load_conviction(args, fold, score=score) if intervention == 'gating' else None

    def _eval(idx):
        empty = torch.tensor([], dtype=torch.long)
        idx = idx if idx is not None else empty
        if intervention == 'masking':
            return _eval_masking(model, data, idx, mask_mode, device, is_auc)
        if intervention == 'gating':
            return _eval_gating_hard(model, data, conviction, idx, device, is_auc)
        return _eval_pruning(model_cls, args, state_dict, data, idx, device, is_auc)

    acc0, conf0, auc0, _ = _eval(None)
    fold_result = {'baseline': {'acc': acc0, 'conf': conf0, 'auc': auc0}}
    for s in STRATEGIES:
        fold_result[s] = {'acc': [], 'conf': [], 'auc': [], 'diagnostics': []}

    for frac in MASK_LEVELS:
        k = max(1, int(frac * len(pool_idx)))
        for strat in STRATEGIES:
            idx = select_candidates(
                strat, k, rng=draw_rng(args['dataset'], fold, strat, frac), **rankings)
            a, c, u, diag = _eval(idx)
            fold_result[strat]['acc'].append(a)
            fold_result[strat]['conf'].append(c)
            fold_result[strat]['auc'].append(u)
            fold_result[strat]['diagnostics'].append(
                {'pct': frac, **diag} if diag is not None else None
            )

    print(f"  fold {fold} done  |  baseline acc={acc0:.3f}  conf={conf0:.3f}"
          + (f"  auc={auc0:.3f}" if is_auc else ""))
    return fold_result

# ── plot ──────────────────────────────────────────────────────────────────────

_INTERVENTION_LABEL = {
    'masking': "feature masking",
    'pruning': "node pruning",
    'gating':  "hard message gating",
}


def _plot(all_results, args, out_path, intervention):
    x_labels = [f"{int(p*100)}%" for p in MASK_LEVELS]
    x_pos    = list(range(len(MASK_LEVELS)))
    n_folds  = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    is_auc   = args.get('dataset') in AUC_DATASETS

    for ax, (metric, title) in zip(axes, [
        ('auc' if is_auc else 'acc',
         'ROC-AUC on test set' if is_auc else 'Accuracy on test set'),
        ('conf', 'Mean confidence on test set'),
    ]):
        base_mu = np.nanmean([r['baseline'].get(metric, float('nan')) for r in all_results])
        ax.axhline(base_mu, color='black', linestyle='--', linewidth=0.9,
                   label=f'No intervention ({base_mu:.3f})')
        for strat, lbl, col, ls, mk in zip(STRATEGIES, LABELS, COLORS, LINESTYLES, MARKERS):
            mat = np.array([r[strat][metric] for r in all_results])
            mu  = np.nanmean(mat, 0)
            se  = np.nanstd(mat,  0) / np.sqrt(n_folds)
            ax.plot(x_pos, mu, label=lbl, color=col, linestyle=ls,
                    marker=mk, linewidth=1.8, markersize=5)
            ax.fill_between(x_pos, mu-se, mu+se, alpha=0.12, color=col)
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("Fraction of train pool intervened on", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)

    tag = map_tag(args)
    score_label = {
        'last':  "X_L conviction",
        'avg':   "avg conviction (layers 1..L)",
        'avg_z': "avg z-scored conviction (layers 1..L)",
    }[args.get('conviction_score', 'last')]
    fig.suptitle(
        f"Conviction {_INTERVENTION_LABEL[intervention]} — {args['dataset']}  "
        f"({args['model']}{tag}, d={args['d']}, {args['layers']} layers)\n"
        f"{score_label}  |  {_INTERVENTION_LABEL[intervention]} (train pool only)  |  full test set  |  "
        f"mean ± SE, {n_folds} folds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    plt.close(fig)


SOFT_INDICATORS = ['conviction',       'confidence', 'degree']
SOFT_IND_LABELS = ['Conviction',        'Confidence', 'Degree']
SOFT_IND_COLORS = ['purple',            'darkcyan',   'darkorange']


def _plot_soft_gate(all_results, args, out_path):
    """Baseline-vs-soft-gated bar comparison — no budget axis, since soft
    gating has no candidate selection/sweep, just one number per fold. One
    gated bar per indicator (conviction/confidence/degree), each gating the
    whole train pool by that score with the same low->min_scale recipe."""
    n_folds = len(all_results)
    is_auc  = args.get('dataset') in AUC_DATASETS
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (metric, title) in zip(axes, [
        ('auc' if is_auc else 'acc',
         'ROC-AUC on test set' if is_auc else 'Accuracy on test set'),
        ('conf', 'Mean confidence on test set'),
    ]):
        baseline_vals = np.array([r['baseline'].get(metric, float('nan')) for r in all_results])
        labels = ['No gating']
        means  = [np.nanmean(baseline_vals)]
        ses    = [np.nanstd(baseline_vals) / np.sqrt(n_folds)]
        colors = ['gray']
        for name, lbl, col in zip(SOFT_INDICATORS, SOFT_IND_LABELS, SOFT_IND_COLORS):
            vals = np.array([r[f'soft_gate_{name}'].get(metric, float('nan')) for r in all_results])
            means.append(np.nanmean(vals))
            ses.append(np.nanstd(vals) / np.sqrt(n_folds))
            labels.append(f'Soft gate\n({lbl})')
            colors.append(col)
        ax.bar(labels, means, yerr=ses, color=colors, capsize=6, alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    tag = map_tag(args)
    score_label = {
        'last':  "X_L conviction",
        'avg':   "avg conviction (layers 1..L)",
        'avg_z': "avg z-scored conviction (layers 1..L)",
    }[args.get('conviction_score', 'last')]
    fig.suptitle(
        f"Conviction soft gating — {args['dataset']}  "
        f"({args['model']}{tag}, d={args['d']}, {args['layers']} layers)\n"
        f"{score_label}  |  entire train pool gated  |  full test set  |  "
        f"mean ± SE, {n_folds} folds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    plt.close(fig)


def _save_table(all_results, args, plot_path, intervention, mode=''):
    """Tabulate exactly what the left-hand panel plots: the headline metric
    (ROC-AUC for the AUC datasets, accuracy otherwise — never both, never
    confidence) per strategy and budget, mean ± SE across folds."""
    n_folds = len(all_results)
    metric  = metric_key(args['dataset'])
    meta    = table_meta(args, intervention, mode)
    rows = [dict(meta, strategy='baseline', pct='',
                 **dict(zip(('mean', 'se'),
                            mean_se([r['baseline'].get(metric, float('nan'))
                                     for r in all_results], n_folds))),
                 n_folds=n_folds)]
    for strat, lbl in zip(STRATEGIES, LABELS):
        for i, frac in enumerate(MASK_LEVELS):
            mu, se = mean_se([r[strat][metric][i] for r in all_results], n_folds)
            rows.append(dict(meta, strategy=strat, pct=frac,
                             mean=mu, se=se, n_folds=n_folds))
    write_table(rows, tables_path(plot_path), ['strategy', 'pct', 'mean', 'se', 'n_folds'])
    _save_folds_table(all_results, args, plot_path, intervention, mode)


def _save_folds_table(all_results, args, plot_path, intervention, mode=''):
    """The per-fold values behind _save_table's means, one row per (fold,
    strategy, budget). Written alongside the aggregated table because that one
    collapses the fold axis before any downstream script sees it, making
    fold-level statistics unrecoverable without re-running the whole sweep.
    'fold' indexes completed folds, so it skips any that were missing weights."""
    metric = metric_key(args['dataset'])
    meta   = table_meta(args, intervention, mode)
    rows   = []
    for fold, r in enumerate(all_results):
        rows.append(dict(meta, fold=fold, strategy='baseline', pct='',
                         value=r['baseline'].get(metric, float('nan'))))
        for strat in STRATEGIES:
            for i, frac in enumerate(MASK_LEVELS):
                rows.append(dict(meta, fold=fold, strategy=strat, pct=frac,
                                 value=r[strat][metric][i]))
    write_table(rows, tables_path(plot_path).replace('.csv', '_folds.csv'),
                ['fold', 'strategy', 'pct', 'value'])


def _save_table_soft_gate(all_results, args, plot_path):
    """One row per indicator (plus the ungated baseline) — the bar heights of
    the soft-gating figure, headline metric only."""
    n_folds = len(all_results)
    metric  = metric_key(args['dataset'])
    meta    = table_meta(args, 'gating', 'soft')
    rows    = []
    for name, key in [('baseline', 'baseline')] + [(n, f'soft_gate_{n}') for n in SOFT_INDICATORS]:
        mu, se = mean_se([r[key].get(metric, float('nan')) for r in all_results], n_folds)
        rows.append(dict(meta, indicator=name, mean=mu, se=se, n_folds=n_folds))
    write_table(rows, tables_path(plot_path), ['indicator', 'mean', 'se', 'n_folds'])
    _save_folds_table_soft_gate(all_results, args, plot_path)


def _save_folds_table_soft_gate(all_results, args, plot_path):
    """The per-fold values behind _save_table_soft_gate's means, one row per
    (fold, indicator) — same rationale as _save_folds_table: the aggregated
    file collapses the fold axis, so keep it here or it is gone. No budget/pct
    column, unlike the other folds tables: soft gating has no candidate
    selection or sweep, just one value per fold per indicator."""
    metric = metric_key(args['dataset'])
    meta   = table_meta(args, 'gating', 'soft')
    rows   = []
    for fold, r in enumerate(all_results):
        rows.append(dict(meta, fold=fold, indicator='baseline',
                         value=r['baseline'].get(metric, float('nan'))))
        for name in SOFT_INDICATORS:
            rows.append(dict(meta, fold=fold, indicator=name,
                             value=r[f'soft_gate_{name}'].get(metric, float('nan'))))
    write_table(rows, tables_path(plot_path).replace('.csv', '_folds.csv'),
                ['fold', 'indicator', 'value'])


def _save_pruning_diagnostics(all_results, args, out_path, strategies=None, levels=None):
    """Alessio's required table: for each (strategy, budget), pct nodes/edges
    remaining, # connected components, class balance per split, split
    composition drift, giant component size — averaged across folds — PLUS the
    headline metric (ROC-AUC for the AUC datasets, accuracy otherwise) so the
    structural damage and the performance drop can be read off one table.
    Every pre-existing field is retained; the metric and provenance columns are
    additions. Written to the Conviction_experiments_tables/ mirror.

    strategies/levels default to this module's STRATEGIES/MASK_LEVELS;
    run_conviction_experiments.py passes its own (shorter) retrain lists."""
    strategies = strategies if strategies is not None else STRATEGIES
    levels     = levels     if levels     is not None else MASK_LEVELS
    metric     = metric_key(args['dataset'])
    n_folds    = len(all_results)
    meta       = table_meta(args, 'pruning', '')
    table = []
    for strat in strategies:
        for i, frac in enumerate(levels):
            entries = [r[strat]['diagnostics'][i] for r in all_results
                       if r[strat]['diagnostics'][i] is not None]
            if not entries:
                continue
            # diagnostics carry their own budget; keep them the authority so a
            # mismatch surfaces instead of silently pairing the wrong rows.
            assert all(e['pct'] == frac for e in entries), \
                f"budget mismatch for {strat}: expected {frac}"
            mu, se = mean_se([r[strat][metric][i] for r in all_results], n_folds)
            row = dict(meta, strategy=strat, pct=frac, n_folds=len(entries),
                       mean=mu, se=se)
            for key in ('pct_nodes_remaining', 'pct_edges_remaining',
                        'n_connected_components', 'giant_component_pct'):
                row[key] = float(np.mean([e[key] for e in entries]))
            row['class_balance_last_fold']  = entries[-1]['class_balance']
            row['split_pct_before_last_fold'] = entries[-1]['split_pct_before']
            row['split_pct_after_last_fold']  = entries[-1]['split_pct_after']
            table.append(row)
    out = tables_path(out_path, ext='.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"  Saved pruning diagnostics table to {out}")

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--normalised', required=True, choices=['true', 'false'])
    parser.add_argument('--map_type',   default='identity',
                        help='For Joint models: identity | general | diag | bundle')
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--dataset',    default=None,
                        help='Restrict the run to one manifest dataset. Use this to shard '
                             'across SLURM jobs — pruning in particular rebuilds a model '
                             'per evaluation and rarely finishes the whole manifest in one '
                             'time window.')
    parser.add_argument('--conviction_score', default='last', choices=['last', 'avg', 'avg_z'],
                        help='last = X_L norm only; avg = mean norm over post-diffusion layers 1..L')
    parser.add_argument('--intervention', default='masking', choices=['masking', 'pruning', 'gating'],
                        help='masking = zero/mean-replace features; pruning = remove nodes+edges; '
                             'gating = attenuate/zero selected nodes\' outgoing messages')
    parser.add_argument('--mask_mode', default='zero', choices=['zero', 'mean'],
                        help='only used when --intervention masking')
    parser.add_argument('--gate_mode', default='hard', choices=['hard', 'soft'],
                        help='only used when --intervention gating: hard = zero the selected '
                             'candidates\' messages (budget sweep, like masking/pruning); '
                             'soft = continuously attenuate every train node\'s message by its '
                             'own conviction (no budget sweep, one point per fold)')
    cli = parser.parse_args()

    device = torch.device(f'cuda:{cli.cuda}' if torch.cuda.is_available() else 'cpu')

    if cli.model in JOINT_MODELS:
        model_tag ='_first_maps_identity' if cli.map_type == 'identity' \
                  else f"_first_maps_{TYPE_SHORT.get(cli.map_type, cli.map_type)}"
    else:
        model_tag =''

    out_dir_parts = [ROOT, 'quick_analysis', 'Conviction_experiments', 'forward_experiment',
                     cli.intervention]
    if cli.intervention == 'gating':
        out_dir_parts.append(cli.gate_mode)
    out_dir_parts += [cli.conviction_score, cli.model, f"normalised_{cli.normalised}"]
    out_dir = os.path.join(*out_dir_parts)
    os.makedirs(out_dir, exist_ok=True)

    gate_info = f"  gate_mode={cli.gate_mode}" if cli.intervention == 'gating' else ''
    print(f"Reading manifest for: model={cli.model}{model_tag}  normalised={cli.normalised}  "
          f"intervention={cli.intervention}{gate_info}"
          + (f"  dataset={cli.dataset}" if cli.dataset else "  (all datasets)"))
    experiments = load_manifest_experiments(cli.model, cli.normalised, model_tag,
                                            dataset_filter=cli.dataset)
    if not experiments:
        print("No experiments found in the manifest. Have you added a line to "
              "quick_analysis/conviction_manifest.txt for this model/normalised combo?"); sys.exit(1)

    print(f"\nFound {len(experiments)} experiment(s).")
    # No module-level generator: each randomised draw builds its own from
    # draw_rng(dataset, fold, strategy, budget), so results no longer depend on
    # how many datasets/folds ran first (see draw_rng in utils/conviction.py).
    MODEL_CLASSES = model_classes()
    from utils.heterophilic import get_dataset

    for targs in experiments:
        tag  = map_tag(targs)
        Norm = 'True' if targs.get('normalised') else 'False'
        n_folds = targs.get('folds', targs['_n_folds_found'])
        print(f"\n{'='*60}\n  {targs['dataset']}  |  {targs['model']}{tag}  "
              f"d={targs['d']}  L={targs['layers']}  folds={n_folds}\n{'='*60}")

        if targs['model'] not in MODEL_CLASSES:
            print(f"  Unknown model '{targs['model']}' — skipping"); continue

        class _FakeArgs:
            pass
        fake = _FakeArgs()
        for k, v in targs.items(): setattr(fake, k, v)
        try:
            dataset = get_dataset(targs['dataset'], fake)
        except Exception as e:
            print(f"  Could not load dataset: {e} — skipping"); continue

        adict = dict(targs)
        adict['device']            = device
        adict['input_dim']         = dataset[0].x.shape[1]
        adict['graph_size']        = dataset[0].x.size(0)
        adict['conviction_score']  = cli.conviction_score
        try:    adict['output_dim'] = dataset.num_classes
        except: adict['output_dim'] = int(torch.unique(dataset[0].y).shape[0])
        if adict.get('sheaf_decay') is None:
            adict['sheaf_decay'] = adict['weight_decay']

        model_cls   = MODEL_CLASSES[targs['model']]
        all_results = []
        for fold in range(n_folds):
            result = _run_fold(adict, dataset, model_cls, fold, device,
                               cli.intervention, cli.mask_mode, cli.gate_mode)
            if result is not None:
                all_results.append(result)

        if not all_results:
            print("  No folds completed — skipping plot"); continue

        if cli.intervention == 'masking':
            mode_tag = f"_{cli.mask_mode}"
        elif cli.intervention == 'gating':
            mode_tag = f"_{cli.gate_mode}"
        else:
            mode_tag = ''
        out_path = os.path.join(out_dir,
            f"{targs['dataset']}_{targs['model']}{tag}_d{targs['d']}_{targs['layers']}L_"
            f"Norm_{Norm}_conv-{cli.conviction_score}{mode_tag}_conviction_{cli.intervention}.pdf")

        mode = cli.mask_mode if cli.intervention == 'masking' else (
               cli.gate_mode if cli.intervention == 'gating' else '')
        if cli.intervention == 'gating' and cli.gate_mode == 'soft':
            _plot_soft_gate(all_results, adict, out_path)
            _save_table_soft_gate(all_results, adict, out_path)
        else:
            _plot(all_results, adict, out_path, cli.intervention)
            _save_table(all_results, adict, out_path, cli.intervention, mode)

        if cli.intervention == 'pruning':
            diag_path = out_path.replace('.pdf', '_diagnostics.json')
            _save_pruning_diagnostics(all_results, adict, diag_path)

    print(f"\nDone. Plots in: {out_dir}")

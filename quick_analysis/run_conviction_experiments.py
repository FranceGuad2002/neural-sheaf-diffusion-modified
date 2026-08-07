#!/usr/bin/env python3
"""
Conviction retrain-from-scratch experiment (Alessio's protocol, phase 2) —
manifest-driven runner, companion to conviction_masking_forward.py.

Which experiments run is read from quick_analysis/conviction_manifest.txt (one
line per dataset/model/hyperparameter combo), filtered by --model/--normalised
and optionally --dataset; nothing is discovered by globbing the results/ tree.

Where conviction_masking_forward.py applies an intervention to a FROZEN,
already-trained model and forward-evaluates it, this script applies the same
intervention to the graph/messages and then RETRAINS a fresh model on it from
scratch, to see whether the intervention actually helps or hurts once the
model gets to adapt to the modified graph.

Scope is intentionally narrower than the forward-eval script, since every
point swept here costs a full training run instead of a single forward pass:
  conviction score : last only (X_L norm)              (--conviction_score
                                                          still accepts avg/avg_z
                                                          for the record, just
                                                          not exercised here)
  budgets          : 1%, 5%, 10% of the train pool
  strategies       : low conviction, low degree, low prediction-confidence,
                     random

For each fold, two numbers anchor the plot:
  reference — the ALREADY-TRAINED checkpoint's test accuracy (one forward
              pass, effectively free) — "how does this compare to the model
              that's actually deployed"
  baseline  — a FRESH retrain with no intervention, same recipe/seed scheme
              as every intervened run — "is the intervention's effect bigger
              than ordinary retrain-to-retrain noise" (see conversation: this
              is the real control, since the reference alone can't separate
              the intervention's effect from run-to-run variance)

Output folder:
    quick_analysis/Conviction_experiments/retrain_experiment/<intervention>/<conviction_score>/<model>/normalised_{true|false}/
(each experiment also writes a _results.json with the raw per-fold numbers,
since — unlike the forward script — these are expensive to regenerate; pruning
additionally writes a _diagnostics.json, same table as the forward script.)

Usage (from repo root, one job per dataset is strongly recommended — see
conversation on why sweeping the whole manifest here can blow SLURM time
limits much faster than the forward-eval script):
    python quick_analysis/run_conviction_experiments.py --model GeneralSheaf --normalised true --dataset roman_empire --intervention masking
    python quick_analysis/run_conviction_experiments.py --model GeneralSheaf --normalised true --dataset roman_empire --intervention pruning
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
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from utils.conviction import (
    JOINT_MODELS, TYPE_SHORT, map_tag, weights_path, load_masks,
    load_fold_rankings, select_candidates, draw_rng, model_classes,
    load_manifest_experiments,
    mask_features, prune_nodes, build_message_gate, load_conviction,
    evaluate_checkpoint,
    AUC_DATASETS, metric_key, tables_path, table_meta, write_table, mean_se,
)
from quick_analysis.conviction_masking_forward import _save_pruning_diagnostics
from exp.run import train, test, precompute_sheaf_mappings

# ── constants ─────────────────────────────────────────────────────────────────

RETRAIN_MASK_LEVELS = [0.01, 0.05, 0.10]
RETRAIN_STRATEGIES  = ['low', 'deg_low', 'conf_low', 'random']

STRATEGY_STYLE = {
    'low':      dict(label='Low conv', color='steelblue', marker='o'),
    'deg_low':  dict(label='Low deg',  color='seagreen',  marker='s'),
    'conf_low': dict(label='Low conf', color='teal',      marker='v'),
    'random':   dict(label='Random',   color='gray',      marker='^'),
}

_INTERVENTION_LABEL = {
    'masking': "feature masking",
    'pruning': "node pruning",
    'gating':  "hard message gating",
}

# ── retraining ────────────────────────────────────────────────────────────────

def _retrain(model_cls, args, data, device, gate=None, seed=None):
    """Fresh model + optimizer, early-stopping training loop mirroring
    exp/run.py's run_exp (same optimizer param groups, same stop_strategy /
    early_stopping logic) but with none of run_exp's side-effect saving — no
    norms/laplacians/saliency/checkpoints written to disk. This is a
    lightweight retrain-and-score, not the main training pipeline; run_exp
    itself is untouched and not called from here."""
    if seed is not None:
        torch.manual_seed(seed)

    if args.get('sheaf_init') and args.get('model') in JOINT_MODELS:
        sheaf_init = precompute_sheaf_mappings(data, args['d'], args)
        model = model_cls(data.edge_index, args, sheaf_init=sheaf_init)
    else:
        model = model_cls(data.edge_index, args)
    model = model.to(device)

    sheaf_learner_params, other_params = model.grouped_parameters()
    maps_lr = args['maps_lr'] if args.get('maps_lr') is not None else args['lr']
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'lr': maps_lr, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']},
    ], lr=args['lr'])

    is_auc = args.get('dataset') in AUC_DATASETS
    best_val_acc, best_val_loss = 0.0, float('inf')
    test_acc, test_conf, test_auc = 0.0, float('nan'), float('nan')
    best_epoch, bad_counter = 0, 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data, gate=gate)
        accs, preds, losses, probs = test(model, data, gate=gate)
        train_acc, val_acc, tmp_test_acc = accs
        val_loss = losses[1]

        new_best = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best:
            best_val_acc, best_val_loss = val_acc, val_loss
            test_acc, best_epoch, bad_counter = tmp_test_acc, epoch, 0
            test_conf = probs[2].max(1).values.mean().item()
            if is_auc:
                y_true = data.y[data.test_mask].cpu().numpy()
                try:
                    test_auc = roc_auc_score(y_true, probs[2].numpy()[:, 1])
                except ValueError:
                    test_auc = float('nan')
        else:
            bad_counter += 1
        if bad_counter == args['early_stopping']:
            break

    return {'acc': test_acc, 'conf': test_conf, 'auc': test_auc,
            'val_acc': best_val_acc, 'best_epoch': best_epoch}


def _prepare_intervention(data, args, idx, conviction, intervention, mask_mode, gate_mode, device):
    """Build the (data, args, gate) triple _retrain should train on for one
    candidate set, plus diagnostics (pruning only). Mirrors the forward
    script's _eval_masking/_eval_pruning/_eval_gating_hard dispatch, except
    the modified graph/gate is TRAINED on here instead of forward-evaluated."""
    if intervention == 'masking':
        return mask_features(data, idx, mode=mask_mode), args, None, None
    if intervention == 'pruning':
        new_data, diagnostics = prune_nodes(data, idx)
        new_data = new_data.to(device)
        new_args = dict(args)
        new_args['graph_size'] = new_data.num_nodes
        return new_data, new_args, None, diagnostics
    if intervention == 'gating':
        gate = build_message_gate(data.num_nodes, conviction, idx, mode=gate_mode, device=device)
        return data, args, gate, None
    raise ValueError(f"Unknown intervention: {intervention}")

# ── per-fold experiment ───────────────────────────────────────────────────────

def _run_fold_retrain(args, dataset, model_cls, fold, device, intervention, mask_mode, gate_mode='hard'):
    data  = dataset[0].clone()
    masks = load_masks(args, fold)
    data.train_mask = masks['train_mask']
    data.val_mask   = masks['val_mask']
    data.test_mask  = masks['test_mask']
    data  = data.to(device)
    n     = data.num_nodes

    wp = weights_path(args, fold)
    if not os.path.exists(wp):
        print(f"  fold {fold}: no original checkpoint — skipping"); return None

    is_auc = args.get('dataset') in AUC_DATASETS
    score  = args.get('conviction_score', 'last')
    seed   = args['seed'] * 1000 + fold

    reference = evaluate_checkpoint(args, fold, model_cls, data, device, is_auc)
    baseline  = _retrain(model_cls, args, data, device, gate=None, seed=seed)
    print(f"  fold {fold}  |  reference acc={reference['acc']:.3f}  "
          f"fresh-baseline acc={baseline['acc']:.3f}")

    rankings   = load_fold_rankings(args, fold, data.edge_index, n, score=score, pool='train')
    pool_idx   = rankings['pool_idx']
    conviction = load_conviction(args, fold, score=score) if intervention == 'gating' else None

    fold_result = {'reference': reference, 'baseline': baseline}
    for s in RETRAIN_STRATEGIES:
        fold_result[s] = {'acc': [], 'conf': [], 'auc': [], 'diagnostics': []}

    for frac in RETRAIN_MASK_LEVELS:
        k = max(1, int(frac * len(pool_idx)))
        for strat in RETRAIN_STRATEGIES:
            idx = select_candidates(
                strat, k, rng=draw_rng(args['dataset'], fold, strat, frac), **rankings)
            train_data, train_args, gate, diag = _prepare_intervention(
                data, args, idx, conviction, intervention, mask_mode, gate_mode, device)
            result = _retrain(model_cls, train_args, train_data, device, gate=gate, seed=seed)
            fold_result[strat]['acc'].append(result['acc'])
            fold_result[strat]['conf'].append(result['conf'])
            fold_result[strat]['auc'].append(result['auc'])
            fold_result[strat]['diagnostics'].append(
                {'pct': frac, **diag} if diag is not None else None
            )
            print(f"    {strat:10s} {int(frac*100):>2d}%  acc={result['acc']:.3f}")

    return fold_result

# ── plot / persist ───────────────────────────────────────────────────────────

def _plot(all_results, args, out_path, intervention):
    x_labels = [f"{int(p*100)}%" for p in RETRAIN_MASK_LEVELS]
    x_pos    = list(range(len(RETRAIN_MASK_LEVELS)))
    n_folds  = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    is_auc   = args.get('dataset') in AUC_DATASETS

    for ax, (metric, title) in zip(axes, [
        ('auc' if is_auc else 'acc',
         'ROC-AUC on test set' if is_auc else 'Accuracy on test set'),
        ('conf', 'Mean confidence on test set'),
    ]):
        base_mu = np.nanmean([r['baseline'].get(metric, float('nan')) for r in all_results])
        ref_mu  = np.nanmean([r['reference'].get(metric, float('nan')) for r in all_results])
        ax.axhline(base_mu, color='black', linestyle='--', linewidth=0.9,
                   label=f'Fresh retrain, no intervention ({base_mu:.3f})')
        ax.axhline(ref_mu, color='dimgray', linestyle=':', linewidth=1.2,
                   label=f'Original checkpoint, reference ({ref_mu:.3f})')
        for strat in RETRAIN_STRATEGIES:
            style = STRATEGY_STYLE[strat]
            mat = np.array([r[strat][metric] for r in all_results])
            mu  = np.nanmean(mat, 0)
            se  = np.nanstd(mat,  0) / np.sqrt(n_folds)
            ax.plot(x_pos, mu, label=style['label'], color=style['color'],
                    marker=style['marker'], linewidth=1.8, markersize=5)
            ax.fill_between(x_pos, mu-se, mu+se, alpha=0.12, color=style['color'])
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("Fraction of train pool intervened on", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)

    tag = map_tag(args)
    fig.suptitle(
        f"Conviction retrain — {_INTERVENTION_LABEL[intervention]} — {args['dataset']}  "
        f"({args['model']}{tag}, d={args['d']}, {args['layers']} layers)\n"
        f"X_L conviction  |  {_INTERVENTION_LABEL[intervention]}, retrain-from-scratch (train pool only)  |  "
        f"full test set  |  mean ± SE, {n_folds} folds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    plt.close(fig)


def _save_table(all_results, args, plot_path, intervention, mode=''):
    """Headline metric only (ROC-AUC for the AUC datasets, accuracy otherwise;
    no confidence), mean ± SE across folds, mirrored into the
    Conviction_experiments_tables/ tree. Both horizontal reference lines of the
    figure are included as their own rows."""
    n_folds = len(all_results)
    metric  = metric_key(args['dataset'])
    meta    = table_meta(args, intervention, mode)
    rows    = []
    for name in ('reference', 'baseline'):
        mu, se = mean_se([r[name].get(metric, float('nan')) for r in all_results], n_folds)
        rows.append(dict(meta, strategy=name, pct='', mean=mu, se=se, n_folds=n_folds))
    for strat in RETRAIN_STRATEGIES:
        for i, frac in enumerate(RETRAIN_MASK_LEVELS):
            mu, se = mean_se([r[strat][metric][i] for r in all_results], n_folds)
            rows.append(dict(meta, strategy=strat, pct=frac, mean=mu, se=se, n_folds=n_folds))
    write_table(rows, tables_path(plot_path), ['strategy', 'pct', 'mean', 'se', 'n_folds'])


def _save_json(all_results, out_path):
    """Raw per-fold/strategy/budget numbers. Unlike the forward script, each
    point here cost a full retrain, so the numbers are persisted here, not
    just plotted."""
    folds = []
    for r in all_results:
        entry = {'reference': r['reference'], 'baseline': r['baseline']}
        for strat in RETRAIN_STRATEGIES:
            entry[strat] = {k: r[strat][k] for k in ('acc', 'conf', 'auc')}
        folds.append(entry)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'mask_levels': RETRAIN_MASK_LEVELS, 'strategies': RETRAIN_STRATEGIES,
                   'folds': folds}, f, indent=2)
    print(f"  Saved raw results to {out_path}")

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--normalised', required=True, choices=['true', 'false'])
    parser.add_argument('--dataset',    default=None,
                        help='Restrict the run to one manifest dataset. Strongly recommended: '
                             'run one SLURM job per dataset, since every point here costs a '
                             'full retrain instead of a single forward pass.')
    parser.add_argument('--map_type',   default='identity',
                        help='For Joint models: identity | general | diag | bundle')
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--conviction_score', default='last', choices=['last', 'avg', 'avg_z'],
                        help='Only "last" is exercised by this reduced-scope retrain sweep; '
                             'avg/avg_z are accepted for future use.')
    parser.add_argument('--intervention', default='masking', choices=['masking', 'pruning', 'gating'])
    parser.add_argument('--mask_mode', default='zero', choices=['zero', 'mean'],
                        help='only used when --intervention masking')
    parser.add_argument('--gate_mode', default='hard', choices=['hard'],
                        help='only used when --intervention gating; soft gating (whole train '
                             'pool, no budget sweep) is not implemented for retraining yet')
    cli = parser.parse_args()

    device = torch.device(f'cuda:{cli.cuda}' if torch.cuda.is_available() else 'cpu')

    if cli.model in JOINT_MODELS:
        model_tag ='_first_maps_identity' if cli.map_type == 'identity' \
                  else f"_first_maps_{TYPE_SHORT.get(cli.map_type, cli.map_type)}"
    else:
        model_tag =''

    out_dir_parts = [ROOT, 'quick_analysis', 'Conviction_experiments', 'retrain_experiment',
                     cli.intervention]
    if cli.intervention == 'gating':
        out_dir_parts.append(cli.gate_mode)
    out_dir_parts += [cli.conviction_score, cli.model, f"normalised_{cli.normalised}"]
    out_dir = os.path.join(*out_dir_parts)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading manifest for: model={cli.model}{model_tag}  normalised={cli.normalised}  "
          f"intervention={cli.intervention}"
          + (f"  dataset={cli.dataset}" if cli.dataset else "  (all datasets — consider --dataset)"))
    experiments = load_manifest_experiments(cli.model, cli.normalised, model_tag, dataset_filter=cli.dataset)
    if not experiments:
        print("No experiments found in the manifest. Have you added a line to "
              "quick_analysis/conviction_manifest.txt for this model/normalised combo?"); sys.exit(1)

    print(f"\nFound {len(experiments)} experiment(s).")
    # No module-level generator: each randomised draw builds its own from
    # draw_rng(dataset, fold, strategy, budget), so this script and the
    # forward-eval one agree despite sweeping different strategy/budget lists.
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
            result = _run_fold_retrain(adict, dataset, model_cls, fold, device,
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
        out_stem = os.path.join(out_dir,
            f"{targs['dataset']}_{targs['model']}{tag}_d{targs['d']}_{targs['layers']}L_"
            f"Norm_{Norm}_conv-{cli.conviction_score}{mode_tag}_conviction_retrain_{cli.intervention}")

        _mode = cli.mask_mode if cli.intervention == 'masking' else (
                cli.gate_mode if cli.intervention == 'gating' else '')
        _plot(all_results, adict, out_stem + '.pdf', cli.intervention)
        _save_table(all_results, adict, out_stem + '.pdf', cli.intervention, _mode)
        _save_json(all_results, out_stem + '_results.json')

        if cli.intervention == 'pruning':
            _save_pruning_diagnostics(all_results, adict, out_stem + '_diagnostics.json',
                                       strategies=RETRAIN_STRATEGIES,
                                       levels=RETRAIN_MASK_LEVELS)

    print(f"\nDone. Plots in: {out_dir}")

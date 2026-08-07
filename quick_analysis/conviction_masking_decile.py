#!/usr/bin/env python3
"""
Conviction decile experiment — manifest-driven runner.

Which experiments run is read from quick_analysis/conviction_manifest.txt (one
line per dataset/model/hyperparameter combo), filtered by --model/--normalised
and optionally --dataset; nothing is discovered by globbing the results/ tree.

Complements conviction_masking_forward.py. Where that script sweeps *cumulative*
budgets (mask the top-k% all at once, k = 1..30%), this one runs 10 *independent
single-bin* experiments per fold: sort the train pool by a score, split it into
equal bins, and intervene on exactly one bin at a time. Every bin has the same
size, so the resulting curve is a like-for-like statement about where in the
score distribution the model's dependence actually lives — a calibrated
importance profile rather than a high/low contrast.

Three ranking criteria are run side by side on the same axes, since the whole
point is the comparison (is conviction a better importance score than simpler
alternatives?):
  conviction — the learned sheaf score (--conviction_score picks last/avg/avg_z)
  confidence — max-softmax prediction confidence, same best-epoch checkpoint
  degree     — plain node degree (topological baseline)
In all three, bin 1 = lowest score, bin 10 = highest.

Interventions (--intervention), all on frozen weights, no retraining:
  masking — zero (or mean-replace, via --mask_mode) the bin's input features
  gating  — zero the bin's OUTGOING messages (hard gating)
  pruning — remove the bin's nodes + incident edges and forward-pass on the
            smaller graph. Costlier (a model is rebuilt per bin, since
            edge_index/graph_size are baked in at construction) and harder to
            read: it confounds losing the bin's features with the structural
            damage of deleting its edges. That confound is worst for the degree
            criterion, where high-degree bins delete far more edges than
            low-degree ones, so the table carries pct_edges_remaining and the
            component statistics per bin — read the curve against them.
Soft gating has no analogue here: it continuously attenuates the entire pool
rather than selecting a subset, so there is nothing to bin.

Candidates come from TRAIN nodes only, so the test set used for evaluation is
identical across every bin, criterion and budget — and bin assignment reuses
load_fold_rankings, so it is tie-broken exactly like the forward experiment.

Output folder:
    quick_analysis/Conviction_experiments/decile_experiment/<intervention>/<conviction_score>/<model>/normalised_{true|false}/
with a CSV of the per-decile mean/SE table written alongside the PDF.

Usage (from repo root):
    python quick_analysis/conviction_masking_decile.py --model GeneralSheaf --normalised true
    python quick_analysis/conviction_masking_decile.py --model GeneralSheaf --normalised true \
        --intervention gating --conviction_score avg_z --dataset cora
"""

import sys
import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.conviction import (
    JOINT_MODELS, TYPE_SHORT, map_tag, weights_path, load_masks,
    load_fold_rankings, model_classes, load_manifest_experiments,
    AUC_DATASETS, metric_key, tables_path, table_meta, write_table, mean_se,
)
# Reused rather than reimplemented so the decile and forward experiments apply
# byte-identical interventions (same masking semantics, same gate construction,
# same test-node exclusion).
from conviction_masking_forward import _eval_masking, _eval_gating_hard, _eval_pruning

# ── constants ─────────────────────────────────────────────────────────────────

N_BINS_DEFAULT = 10

# Ascending rankings from load_fold_rankings: index 0 = lowest score, so bin 1
# is always the bottom of the distribution for every criterion.
CRITERIA   = ['conviction',  'confidence', 'degree']
CRIT_RANK  = {'conviction': 'rank_conv_lo',
              'confidence': 'rank_conf_lo',
              'degree':     'rank_deg_lo'}
CRIT_LABEL = {'conviction': 'Conviction', 'confidence': 'Confidence', 'degree': 'Degree'}
# Same palette as the soft-gating plot, so the indicators read consistently
# across every figure in the conviction study.
CRIT_COLOR = {'conviction': 'purple', 'confidence': 'darkcyan', 'degree': 'darkorange'}
CRIT_MARKER = {'conviction': 'o', 'confidence': 's', 'degree': 'D'}

_INTERVENTION_LABEL = {'masking': "feature masking", 'gating': "hard message gating",
                       'pruning': "node pruning"}

# Structural fields pruning adds to the table, so the metric drop per bin can be
# read against how much of the graph that bin's removal actually destroyed.
PRUNE_DIAG_FIELDS = ['pct_nodes_remaining', 'pct_edges_remaining',
                     'n_connected_components', 'giant_component_pct']

# ── per-fold experiment ───────────────────────────────────────────────────────

def _run_fold(args, dataset, model_cls, fold, device, intervention, mask_mode, n_bins):
    data  = dataset[0].clone()
    masks = load_masks(args, fold)
    data.train_mask = masks['train_mask']
    data.val_mask   = masks['val_mask']
    data.test_mask  = masks['test_mask']
    data = data.to(device)
    n    = data.num_nodes

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

    def _eval(idx):
        """Returns (acc, conf, auc, diagnostics); diagnostics is None for
        everything except pruning, which reports how much of the graph survived."""
        if intervention == 'masking':
            return _eval_masking(model, data, idx, mask_mode, device, is_auc)
        if intervention == 'pruning':
            # graph_size/edge_index are baked in at construction, so pruning
            # rebuilds a model for the smaller graph and reloads the SAME
            # frozen weights — no retraining, just a costlier forward pass.
            return _eval_pruning(model_cls, args, state_dict, data, idx, device, is_auc)
        # hard gating ignores the score vector (it just zeroes the gate), so None
        return _eval_gating_hard(model, data, None, idx, device, is_auc)

    empty = torch.tensor([], dtype=torch.long)
    acc0, conf0, auc0, _ = _eval(empty)
    result = {'baseline': {'acc': acc0, 'conf': conf0, 'auc': auc0}}

    rankings = load_fold_rankings(args, fold, data.edge_index, n, score=score, pool='train')
    for crit in CRITERIA:
        rank = rankings[CRIT_RANK[crit]]
        result[crit] = {'acc': [], 'conf': [], 'auc': [], 'n_nodes': [], 'diagnostics': []}
        # array_split handles pools whose size isn't divisible by n_bins by
        # making the leading bins one node larger — a negligible imbalance.
        for chunk in np.array_split(rank.numpy(), n_bins):
            a, c, u, diag = _eval(torch.as_tensor(chunk, dtype=torch.long))
            result[crit]['acc'].append(a)
            result[crit]['conf'].append(c)
            result[crit]['auc'].append(u)
            result[crit]['n_nodes'].append(len(chunk))
            result[crit]['diagnostics'].append(diag)

    print(f"  fold {fold} done  |  baseline acc={acc0:.3f}  conf={conf0:.3f}"
          + (f"  auc={auc0:.3f}" if is_auc else ""))
    return result

# ── aggregation / output ──────────────────────────────────────────────────────

def _mean_se(all_results, crit, metric, n_folds):
    mat = np.array([r[crit][metric] for r in all_results], dtype=float)
    return np.nanmean(mat, 0), np.nanstd(mat, 0) / np.sqrt(n_folds)


def _plot(all_results, args, out_path, intervention, n_bins):
    """One line per criterion over the bin axis, mean ± SE across folds, with the
    no-intervention baseline as a dashed horizontal reference. Lines rather than
    bars: with three criteria over an ordered axis, 3*n_bins bars is unreadable
    and hides exactly the monotonicity the plot exists to show."""
    n_folds = len(all_results)
    is_auc  = args.get('dataset') in AUC_DATASETS
    x_pos   = list(range(1, n_bins + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (metric, title) in zip(axes, [
        ('auc' if is_auc else 'acc',
         'ROC-AUC on test set' if is_auc else 'Accuracy on test set'),
        ('conf', 'Mean confidence on test set'),
    ]):
        base_mu = np.nanmean([r['baseline'].get(metric, float('nan')) for r in all_results])
        ax.axhline(base_mu, color='black', linestyle='--', linewidth=0.9,
                   label=f'No intervention ({base_mu:.3f})')
        for crit in CRITERIA:
            mu, se = _mean_se(all_results, crit, metric, n_folds)
            ax.plot(x_pos, mu, label=CRIT_LABEL[crit], color=CRIT_COLOR[crit],
                    marker=CRIT_MARKER[crit], linewidth=1.8, markersize=5)
            ax.fill_between(x_pos, mu - se, mu + se, alpha=0.12, color=CRIT_COLOR[crit])
        ax.set_xticks(x_pos)
        ax.set_xlabel(f"Bin of the train pool, ranked low -> high ({n_bins} equal bins)", fontsize=10)
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
        f"Decile {_INTERVENTION_LABEL[intervention]} — {args['dataset']}  "
        f"({args['model']}{tag}, d={args['d']}, {args['layers']} layers)\n"
        f"{score_label}  |  one bin intervened on at a time (train pool only)  |  "
        f"full test set  |  mean ± SE, {n_folds} folds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    plt.close(fig)


def _save_table(all_results, args, plot_path, intervention, mode, n_bins):
    """Per-bin ablation table: the headline metric only (ROC-AUC for the AUC
    datasets, accuracy otherwise — no confidence column), mean ± SE across
    folds, written to the Conviction_experiments_tables/ mirror of plot_path."""
    n_folds = len(all_results)
    metric  = metric_key(args['dataset'])
    meta    = table_meta(args, intervention, mode)
    is_prune   = intervention == 'pruning'
    mu_b, se_b = mean_se([r['baseline'].get(metric, float('nan')) for r in all_results], n_folds)
    rows = [dict(meta, criterion='baseline', bin='', n_nodes=0,
                 mean=mu_b, se=se_b, n_folds=n_folds)]
    for crit in CRITERIA:
        mu, se  = _mean_se(all_results, crit, metric, n_folds)
        n_nodes = np.mean([r[crit]['n_nodes'] for r in all_results], 0)
        for b in range(n_bins):
            row = dict(meta, criterion=crit, bin=b + 1, n_nodes=float(n_nodes[b]),
                       mean=float(mu[b]), se=float(se[b]), n_folds=n_folds)
            if is_prune:
                # Without these, a pruning decile curve is uninterpretable: a bin
                # can look "important" purely because removing it deleted more edges.
                entries = [r[crit]['diagnostics'][b] for r in all_results
                           if r[crit]['diagnostics'][b] is not None]
                for key in PRUNE_DIAG_FIELDS:
                    row[key] = float(np.mean([e[key] for e in entries])) if entries else float('nan')
            rows.append(row)
    write_table(rows, tables_path(plot_path),
                ['criterion', 'bin', 'n_nodes', 'mean', 'se', 'n_folds']
                + (PRUNE_DIAG_FIELDS if is_prune else []))
    _save_folds_table(all_results, args, plot_path, intervention, mode, n_bins)


def _save_folds_table(all_results, args, plot_path, intervention, mode, n_bins):
    """The per-fold values behind _save_table's means, one row per (fold,
    criterion, bin) — same rationale as the forward experiment's folds table:
    the aggregated file collapses the fold axis, so keep it here or it is gone."""
    metric = metric_key(args['dataset'])
    meta   = table_meta(args, intervention, mode)
    rows   = []
    for fold, r in enumerate(all_results):
        rows.append(dict(meta, fold=fold, criterion='baseline', bin='',
                         value=r['baseline'].get(metric, float('nan'))))
        for crit in CRITERIA:
            for b in range(n_bins):
                rows.append(dict(meta, fold=fold, criterion=crit, bin=b + 1,
                                 value=r[crit][metric][b]))
    write_table(rows, tables_path(plot_path).replace('.csv', '_folds.csv'),
                ['fold', 'criterion', 'bin', 'value'])

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--normalised', required=True, choices=['true', 'false'])
    parser.add_argument('--map_type',   default='identity',
                        help='For Joint models: identity | general | diag | bundle')
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--dataset',    default=None,
                        help='Optional: restrict the run to a single manifest dataset')
    parser.add_argument('--conviction_score', default='last', choices=['last', 'avg', 'avg_z'],
                        help='last = X_L norm only; avg = mean norm over post-diffusion '
                             'layers 1..L; avg_z = same, z-scored per layer first')
    parser.add_argument('--intervention', default='masking',
                        choices=['masking', 'gating', 'pruning'],
                        help='masking = zero/mean-replace the bin\'s features; '
                             'gating = zero the bin\'s outgoing messages (hard); '
                             'pruning = remove the bin\'s nodes + incident edges (slower, '
                             'rebuilds the model per bin, and confounds feature loss with '
                             'structural damage — read the edges-remaining column). Soft '
                             'gating has no decile analogue — it gates the whole pool.')
    parser.add_argument('--mask_mode', default='zero', choices=['zero', 'mean'],
                        help='only used when --intervention masking')
    parser.add_argument('--n_bins', type=int, default=N_BINS_DEFAULT,
                        help='number of equal-sized bins (10 = deciles)')
    cli = parser.parse_args()

    device = torch.device(f'cuda:{cli.cuda}' if torch.cuda.is_available() else 'cpu')

    if cli.model in JOINT_MODELS:
        model_tag ='_first_maps_identity' if cli.map_type == 'identity' \
                  else f"_first_maps_{TYPE_SHORT.get(cli.map_type, cli.map_type)}"
    else:
        model_tag =''

    out_dir = os.path.join(ROOT, 'quick_analysis', 'Conviction_experiments',
                           'decile_experiment', cli.intervention, cli.conviction_score,
                           cli.model, f"normalised_{cli.normalised}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading manifest for: model={cli.model}{model_tag}  normalised={cli.normalised}  "
          f"intervention={cli.intervention}  bins={cli.n_bins}")
    experiments = load_manifest_experiments(cli.model, cli.normalised, model_tag,
                                            dataset_filter=cli.dataset)
    if not experiments:
        print("No experiments found in the manifest. Have you added a line to "
              "quick_analysis/conviction_manifest.txt for this model/normalised combo?")
        sys.exit(1)

    print(f"\nFound {len(experiments)} experiment(s).")
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
        adict['device']           = device
        adict['input_dim']        = dataset[0].x.shape[1]
        adict['graph_size']       = dataset[0].x.size(0)
        adict['conviction_score'] = cli.conviction_score
        try:    adict['output_dim'] = dataset.num_classes
        except: adict['output_dim'] = int(torch.unique(dataset[0].y).shape[0])
        if adict.get('sheaf_decay') is None:
            adict['sheaf_decay'] = adict['weight_decay']

        model_cls   = MODEL_CLASSES[targs['model']]
        all_results = []
        for fold in range(n_folds):
            result = _run_fold(adict, dataset, model_cls, fold, device,
                               cli.intervention, cli.mask_mode, cli.n_bins)
            if result is not None:
                all_results.append(result)

        if not all_results:
            print("  No folds completed — skipping plot"); continue

        mode     = {'masking': cli.mask_mode, 'gating': 'hard', 'pruning': ''}[cli.intervention]
        mode_tag = f"_{cli.mask_mode}" if cli.intervention == 'masking' else ''
        stem = (f"{targs['dataset']}_{targs['model']}{tag}_d{targs['d']}_{targs['layers']}L_"
                f"Norm_{Norm}_conv-{cli.conviction_score}{mode_tag}_decile_{cli.intervention}")
        plot_path = os.path.join(out_dir, stem + '.pdf')
        _plot(all_results, adict, plot_path, cli.intervention, cli.n_bins)
        _save_table(all_results, adict, plot_path, cli.intervention, mode, cli.n_bins)

    print(f"\nDone. Plots in: {out_dir}")

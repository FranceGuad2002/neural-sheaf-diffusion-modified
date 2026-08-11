#!/usr/bin/env python3
"""
Conviction heatmap — auto-discovery runner.

For each training checkpoint and diffusion-layer snapshot, computes:
  - Spearman ρ: conviction vs degree / betweenness / confidence / saliency
  - Cohen's d:  correct vs incorrect nodes (train and test split)
All metrics averaged across folds.  Output: 2×3 heatmap PDF.

Output folder:
    quick_analysis/conviction_heatmap/<model>/normalised_{true|false}/

Usage (from repo root):
    python quick_analysis/conviction_heatmap.py --model GeneralSheaf --normalised true
    python quick_analysis/conviction_heatmap.py --model BundleSheaf  --normalised false
"""

import sys
import os
import re
import csv
import glob
import json
import argparse

import numpy as np
import networkx as nx
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr, linregress

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from definitions import ROOT_DIR

# ── constants ─────────────────────────────────────────────────────────────────

_NPY_DATASETS   = {"roman_empire", "minesweeper", "amazon_ratings", "tolokers", "questions"}
_WEBKB_DATASETS = {"cornell", "texas", "wisconsin"}
_JOINT_MODELS   = {'JointSheafParams', 'JointSheafParamsAlt'}
_TYPE_SHORT     = {'diagonal': 'diag', 'bundle': 'bundle', 'general': 'general'}
EXCLUDE         = {'synthetic_exp'}

# Internal buffer key -> name used in the per-fold CSV and in the summary table.
OBSERVABLE_NAME = {
    'deg':     'degree',
    'bet':     'betweenness',
    'conf':    'confidence',
    'sal':     'saliency',
    'd_train': 'cohens_d_train',
    'd_test':  'cohens_d_test',
}

HEATMAP_SPECS = [
    ('rho_deg_mean',  'Spearman ρ\n(conviction vs degree)',      'RdBu_r', -1.0, 0, 1.0),
    ('rho_bet_mean',  'Spearman ρ\n(conviction vs betweenness)', 'RdBu_r', -1.0, 0, 1.0),
    ('rho_conf_mean', 'Spearman ρ\n(conviction vs confidence)',  'RdBu_r', -1.0, 0, 1.0),
    ('rho_sal_mean',  'Spearman ρ\n(conviction vs saliency)',    'RdBu_r', -1.0, 0, 1.0),
    ('d_train_mean',  "|Cohen's d|\n(accuracy, train set)",       'YlOrRd',  0.0, 0.5, 1.5),
    ('d_test_mean',   "|Cohen's d|\n(accuracy, test set)",        'YlOrRd',  0.0, 0.5, 1.5),
]

# ── path helpers ──────────────────────────────────────────────────────────────

def _map_tag(args):
    if args.get('model') not in _JOINT_MODELS:
        return ""
    if not args.get('learn_first_maps', False):
        return "_first_maps_identity"
    short = _TYPE_SHORT.get(args.get('learnt_map_type', 'general'), 'general')
    return f"_first_maps_{short}"


def _base_dir(result_type, args):
    return os.path.join(
        ROOT_DIR, 'results', result_type,
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    )


def _pt_filename(args, fold):
    return f"{args['model']}_{args['dataset']}{_map_tag(args)}_fold{fold}_seed{args['seed']}.pt"


def _ckpt_path(result_type, args, fold, ckpt_name):
    fn = _pt_filename(args, fold)
    if ckpt_name == 'last':
        return os.path.join(_base_dir(result_type, args), fn)
    return os.path.join(_base_dir(result_type, args), 'checkpoints', ckpt_name, fn)


def _preds_path(args, fold):
    return os.path.join(_base_dir('node_preds', args), _pt_filename(args, fold))

# ── checkpoint discovery ──────────────────────────────────────────────────────

def _ckpt_sort_key(name):
    if name == 'last':        return (2, 0)
    if name == 'best-epoch':  return (1, 0)
    try:                      return (0, int(name.split('-')[1]))
    except (IndexError, ValueError): return (3, 0)


def _available_checkpoints(args):
    ckpt_root = os.path.join(_base_dir('node_norms', args), 'checkpoints')
    names = [os.path.basename(d) for d in glob.glob(os.path.join(ckpt_root, '*'))
             if os.path.isdir(d)]
    if os.path.exists(os.path.join(_base_dir('node_norms', args), _pt_filename(args, 0))):
        names.append('last')
    return sorted(names, key=_ckpt_sort_key)

# ── graph topology ────────────────────────────────────────────────────────────

def _build_topology(dataset_name):
    data_root = os.path.join(ROOT_DIR, 'datasets')

    if dataset_name in _NPY_DATASETS:
        raw = np.load(os.path.join(data_root, dataset_name, 'raw', 'edges.npy'))
        src, dst = raw[:, 0].astype(int), raw[:, 1].astype(int)
        n = int(np.load(os.path.join(data_root, dataset_name, 'raw', 'node_features.npy')).shape[0])
    elif dataset_name in _WEBKB_DATASETS:
        raw = np.loadtxt(os.path.join(data_root, dataset_name, 'raw', 'out1_graph_edges.txt'),
                         dtype=int, skiprows=1)
        src, dst = raw[:, 0], raw[:, 1]
        with open(os.path.join(data_root, dataset_name, 'raw', 'out1_node_feature_label.txt')) as f:
            n = sum(1 for _ in f) - 1
    else:
        from torch_geometric.datasets import Planetoid
        import torch_geometric.transforms as T
        ds = Planetoid(root=data_root, name=dataset_name, transform=T.NormalizeFeatures())
        ei = ds[0].edge_index.numpy()
        src, dst, n = ei[0], ei[1], ds[0].num_nodes

    edges = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(src, dst) if u != v}
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    nodelist = list(range(n))

    deg = np.array([G.degree(u) for u in nodelist], dtype=float)
    k   = min(500, n)
    print(f"  Computing betweenness (k={k}) …")
    bet = nx.betweenness_centrality(G, k=k, seed=42)
    bet = np.array([bet[u] for u in nodelist], dtype=float)
    print(f"  Graph: {n} nodes, {G.number_of_edges()} edges")
    return deg, bet

# ── analysis ──────────────────────────────────────────────────────────────────

def _cohens_d(conv_subset, correct_mask):
    c_ok, c_fail = conv_subset[correct_mask], conv_subset[~correct_mask]
    n1, n2 = len(c_ok), len(c_fail)
    if n1 < 2 or n2 < 2:
        return float('nan')
    s_p = np.sqrt(((n1-1)*c_ok.std()**2 + (n2-1)*c_fail.std()**2) / (n1+n2-2))
    return (c_ok.mean() - c_fail.mean()) / s_p if s_p > 0 else float('nan')


def _compute(args, node_deg, node_bet):
    """Returns (results, checkpoint_names, fold_rows).

    `results` holds the fold-averaged grids the heatmap plots. `fold_rows` keeps
    the SAME quantities one row per (checkpoint, layer, fold, observable) before
    any averaging, so that a cross-dataset table can average folds within a
    dataset and take its spread across datasets, exactly as the intervention
    tables do. The plot cannot do this for us: it averages the folds away."""
    checkpoint_names = _available_checkpoints(args)
    num_snapshots    = args['layers'] + 1
    n_folds          = args.get('folds', 10)
    results          = {}
    fold_rows        = []

    for ckpt_name in checkpoint_names:
        print(f"\n  --- {ckpt_name} ---")
        bufs      = {k: [] for k in ('deg', 'bet', 'conf', 'sal', 'd_train', 'd_test')}
        fold_ids  = []

        for fold in range(n_folds):
            np_path = _ckpt_path('node_norms',    args, fold, ckpt_name)
            cp_path = _ckpt_path('node_conf',     args, fold, ckpt_name)
            sp_path = _ckpt_path('node_saliency', args, fold, ckpt_name)
            ap_path = _ckpt_path('node_accuracy', args, fold, ckpt_name)
            pp_path = _preds_path(args, fold)

            if not os.path.exists(np_path):
                print(f"    fold {fold} — norms missing, skipped")
                continue

            norms_dict = torch.load(np_path, weights_only=False)
            conf_vec = torch.load(cp_path, weights_only=False).numpy() if os.path.exists(cp_path) else None
            sal_vec  = torch.load(sp_path, weights_only=False).numpy() if os.path.exists(sp_path) else None
            acc_vec  = torch.load(ap_path, weights_only=False).numpy().astype(bool) if os.path.exists(ap_path) else None

            train_mask = test_mask = None
            if os.path.exists(pp_path) and acc_vec is not None:
                _p = torch.load(pp_path, weights_only=False)
                train_mask, test_mask = _p['train_mask'].numpy(), _p['test_mask'].numpy()

            row = {k: [] for k in bufs}
            ok  = True
            for t in range(num_snapshots):
                if t not in norms_dict:
                    print(f"    fold {fold} — snapshot {t} missing, skipped"); ok = False; break
                nt = norms_dict[t].numpy()
                row['deg'].append(spearmanr(nt, node_deg).statistic)
                row['bet'].append(spearmanr(nt, node_bet).statistic)
                row['conf'].append(spearmanr(nt, conf_vec).statistic if conf_vec is not None else float('nan'))
                row['sal'].append(spearmanr(nt,  sal_vec).statistic if sal_vec  is not None else float('nan'))
                if acc_vec is not None and train_mask is not None:
                    row['d_train'].append(abs(_cohens_d(nt[train_mask], acc_vec[train_mask])))
                    row['d_test'].append( abs(_cohens_d(nt[test_mask],  acc_vec[test_mask])))
                else:
                    row['d_train'].append(float('nan')); row['d_test'].append(float('nan'))

            if ok:
                for k in bufs: bufs[k].append(row[k])
                fold_ids.append(fold)
                print(f"    fold {fold} done")

        n_avail = len(bufs['deg'])
        if n_avail == 0:
            print("  No folds available — skipping."); continue

        for i, fold in enumerate(fold_ids):
            for k, observable in OBSERVABLE_NAME.items():
                for t in range(num_snapshots):
                    fold_rows.append({
                        'dataset': args['dataset'], 'model': args['model'],
                        'normalised': 'true' if args.get('normalised') else 'false',
                        'd': args['d'], 'layers': args['layers'],
                        'checkpoint': ckpt_name, 'layer': t, 'fold': fold,
                        'observable': observable, 'value': bufs[k][i][t],
                    })

        a = {k: np.array(bufs[k]) for k in bufs}
        results[ckpt_name] = {
            'rho_deg_mean':  a['deg'].mean(0),   'rho_deg_std':  a['deg'].std(0),
            'rho_bet_mean':  a['bet'].mean(0),   'rho_bet_std':  a['bet'].std(0),
            'rho_conf_mean': np.nanmean(a['conf'],    0), 'rho_conf_std': np.nanstd(a['conf'],    0),
            'rho_sal_mean':  np.nanmean(a['sal'],     0), 'rho_sal_std':  np.nanstd(a['sal'],     0),
            'd_train_mean':  np.nanmean(a['d_train'], 0), 'd_train_std':  np.nanstd(a['d_train'], 0),
            'd_test_mean':   np.nanmean(a['d_test'],  0), 'd_test_std':   np.nanstd(a['d_test'],  0),
            'n_folds': n_avail,
        }
        r = results[ckpt_name]
        print(f"  -> {n_avail} fold(s) | deg ρ@X_L={r['rho_deg_mean'][-1]:.3f}  "
              f"conf ρ@X_L={r['rho_conf_mean'][-1]:.3f}  "
              f"d_test@X_L={r['d_test_mean'][-1]:.3f}")

    return results, checkpoint_names, fold_rows


def _write_fold_csv(fold_rows, path):
    """Per-fold observational values, the input to conviction_observational_table.py."""
    if not fold_rows:
        print("  (no fold rows to write)"); return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        w.writeheader(); w.writerows(fold_rows)
    print(f"  -> wrote {len(fold_rows)} fold rows to {path}")

# ── plot ──────────────────────────────────────────────────────────────────────

def _plot(results, checkpoint_names, args, out_path):
    num_snapshots   = args['layers'] + 1
    snapshot_labels = [f"X_{t}" for t in range(num_snapshots)]
    valid   = [c for c in checkpoint_names if c in results]
    n_ckpts = len(valid)
    if n_ckpts == 0:
        print("  Nothing to plot."); return

    mats = {}
    for key, *_ in HEATMAP_SPECS:
        mat = np.full((n_ckpts, num_snapshots), np.nan)
        for i, ckpt in enumerate(valid):
            mat[i] = results[ckpt][key]
        mats[key] = mat

    cell_w, cell_h = 0.85, 0.55
    fig_w = 3 * num_snapshots * cell_w + 2*2.5 + 1.5
    fig_h = 2 * n_ckpts       * cell_h + 1.5   + 1.0
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_h))
    tag = _map_tag(args)

    for ax, (key, title, cmap, vmin, vcenter, vmax) in zip(axes.flat, HEATMAP_SPECS):
        mat = mats[key]
        im  = ax.imshow(mat, aspect='auto', cmap=cmap,
                        norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
                        interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=6)
        ax.set_xlabel("Diffusion layer", fontsize=12)
        ax.set_xticks(range(num_snapshots)); ax.set_xticklabels(snapshot_labels, fontsize=12)
        ax.set_yticks(range(n_ckpts));       ax.set_yticklabels(valid, fontsize=12)
        max_abs = max(abs(vmin), abs(vmax))
        for i in range(n_ckpts):
            for j in range(num_snapshots):
                v = mat[i, j]
                if not np.isnan(v):
                    color = 'white' if abs(v)/max_abs > 0.6 else 'black'
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=12, color=color)
                elif key in ('d_train_mean', 'd_test_mean'):
                    ax.text(j, i, "100%", ha='center', va='center', fontsize=10,
                            color='#666666', style='italic')

    fig.suptitle(
        f"Conviction heatmap — {args['dataset']}  ({args['model']}{tag}, d={args['d']}, {args['layers']} layers)\n"
        f"rows = training checkpoint  |  cols = diffusion layer  |  cells = mean across folds",
        fontsize=25, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    plt.close(fig)

# ── scatter helpers ───────────────────────────────────────────────────────────

def _load_norms_avg(args, ckpt_name='best-epoch'):
    num_snapshots = args['layers'] + 1
    n_folds       = args.get('folds', 10)
    norms_sum     = None
    fold_count    = 0
    for fold in range(n_folds):
        np_path = _ckpt_path('node_norms', args, fold, ckpt_name)
        if not os.path.exists(np_path):
            continue
        norms_dict = torch.load(np_path, weights_only=False)
        if norms_sum is None:
            n_nodes   = norms_dict[0].shape[0]
            norms_sum = np.zeros((num_snapshots, n_nodes))
        for t in range(num_snapshots):
            if t in norms_dict:
                norms_sum[t] += norms_dict[t].numpy()
        fold_count += 1
    if fold_count == 0:
        return None, 0
    return norms_sum / fold_count, fold_count


def _plot_scatter(norms_avg, fold_count, node_deg, args, out_path):
    num_snapshots = args['layers'] + 1
    deg = np.asarray(node_deg, dtype=float)

    fig, axes = plt.subplots(1, num_snapshots,
                             figsize=(3.5 * num_snapshots, 4.0),
                             sharey=False)
    if num_snapshots == 1:
        axes = [axes]

    SCATTER_C = "#7B8EC8"
    LINE_C    = "#4A5899"

    for t, ax in enumerate(axes):
        y     = norms_avg[t]
        valid = np.isfinite(deg) & np.isfinite(y)
        xv, yv = deg[valid], y[valid]

        ax.scatter(xv, yv, color=SCATTER_C, alpha=0.8, s=30,
                   edgecolors="white", linewidths=0.4, rasterized=True)

        slope, intercept, *_ = linregress(xv, yv)
        xs = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(xs, slope * xs + intercept, color=LINE_C, lw=1.6, ls="--", alpha=0.85)

        rho, pval = spearmanr(xv, yv)
        p_str = f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}"
        ax.text(0.05, 0.95,
                f"$\\rho_s = {rho:.3f}$\n$p = {p_str}$",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

        label = "X_0 (input)" if t == 0 else f"X_{t}"
        ax.set_title(label, fontsize=15)
        ax.set_xlabel("Degree", fontsize=15)
        if t == 0:
            ax.set_ylabel("Conviction (L2 norm)", fontsize=9)

    Norm = 'True' if args.get('normalised') else 'False'
    tag  = _map_tag(args)
    fig.suptitle(
        f"Conviction vs Degree — {args['dataset']} | {args['model']}{tag} | "
        f"norm={Norm} | best-epoch | {fold_count} fold(s)",
        fontsize=20, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Scatter saved to {out_path}")
    plt.close(fig)

# ── experiment discovery ──────────────────────────────────────────────────────

def _args_from_path(epochs_dir, model_name, normalised_str):
    from exp.parser import get_parser
    h_dir, l_dir = os.path.dirname(epochs_dir), os.path.dirname(os.path.dirname(epochs_dir))
    d_dir        = os.path.dirname(l_dir)
    dataset_dir  = os.path.dirname(os.path.dirname(os.path.dirname(d_dir)))
    try:
        epochs          = int(os.path.basename(epochs_dir).split('-')[0])
        hidden_channels = int(os.path.basename(h_dir).split('-')[0])
        layers          = int(os.path.basename(l_dir).split('-')[0])
        d               = int(os.path.basename(d_dir).split('-')[-1])
        dataset_name    = os.path.basename(dataset_dir)
    except (ValueError, IndexError):
        return None
    pt_files   = glob.glob(os.path.join(epochs_dir, 'checkpoints', '*', '*.pt'))
    seed_match = re.search(r'_seed(\d+)\.pt$', pt_files[0]) if pt_files else None
    seed       = int(seed_match.group(1)) if seed_match else 43
    defaults   = vars(get_parser().parse_args([]))
    defaults.update({
        'dataset': dataset_name, 'model': model_name, 'd': d, 'layers': layers,
        'hidden_channels': hidden_channels, 'epochs': epochs, 'seed': seed,
        'normalised': normalised_str == 'true', 'deg_normalised': False,
        'folds': 10, 'edge_weights': False,
    })
    return defaults


def _discover(model_name, normalised_str, map_tag):
    norms_root   = os.path.join(ROOT_DIR, 'results', 'node_norms')
    weights_root = os.path.join(ROOT_DIR, 'results', 'model_weights')
    pattern = os.path.join(norms_root, '*', model_name + map_tag,
                           f'normalised-{normalised_str}',
                           'stalk_dim-*', '*-layers', '*-hidden', '*-epochs')
    experiments = []
    for epochs_dir in sorted(glob.glob(pattern)):
        if not glob.glob(os.path.join(epochs_dir, 'checkpoints', '*', '*.pt')):
            continue
        rel       = os.path.relpath(epochs_dir, norms_root)
        args_path = os.path.join(weights_root, rel, 'training_args.json')
        if os.path.exists(args_path):
            with open(args_path) as f:
                targs = json.load(f)
        else:
            targs = _args_from_path(epochs_dir, model_name, normalised_str)
            if targs is None:
                print(f"  Could not parse path: {epochs_dir} — skipping"); continue
            print(f"  [WARNING] training_args.json missing, inferred from path")
        if targs.get('dataset', '') in EXCLUDE:
            continue
        experiments.append(targs)
        print(f"  found: {targs['dataset']}  d={targs['d']}  L={targs['layers']}")
    return experiments

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--normalised', required=True, choices=['true', 'false'])
    parser.add_argument('--map_type',   default='identity',
                        help='For Joint models: identity | general | diag | bundle')
    cli = parser.parse_args()

    if cli.model in _JOINT_MODELS:
        map_tag = '_first_maps_identity' if cli.map_type == 'identity' \
                  else f"_first_maps_{_TYPE_SHORT.get(cli.map_type, cli.map_type)}"
    else:
        map_tag = ''

    out_dir     = os.path.join(ROOT_DIR, 'quick_analysis', 'Norm_Analysis', 'conviction_heatmap',
                               cli.model, f"normalised_{cli.normalised}")
    scatter_dir = os.path.join(ROOT_DIR, 'quick_analysis', 'Norm_Analysis', 'conviction_scatter',
                               cli.model, f"normalised_{cli.normalised}")
    folds_dir   = os.path.join(ROOT_DIR, 'quick_analysis', 'Conviction_experiments_tables',
                               'observational', cli.model, f"normalised_{cli.normalised}")
    os.makedirs(out_dir,     exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)
    os.makedirs(folds_dir,   exist_ok=True)

    print(f"Discovering: model={cli.model}{map_tag}  normalised={cli.normalised}")
    experiments = _discover(cli.model, cli.normalised, map_tag)
    if not experiments:
        print("No experiments found."); sys.exit(1)

    print(f"\nFound {len(experiments)} experiment(s).")
    for targs in experiments:
        tag  = _map_tag(targs)
        Norm = 'True' if targs.get('normalised') else 'False'
        print(f"\n{'='*60}\n  {targs['dataset']}  |  {targs['model']}{tag}  d={targs['d']}  L={targs['layers']}\n{'='*60}")
        deg, bet = _build_topology(targs['dataset'])
        results, ckpt_names, fold_rows = _compute(targs, deg, bet)
        if not results:
            continue
        stem = (f"{targs['dataset']}_{targs['model']}{tag}_d{targs['d']}_{targs['layers']}L_"
                f"Norm_{Norm}")
        _plot(results, ckpt_names, targs,
              os.path.join(out_dir, f"{stem}_conviction_heatmap.pdf"))
        _write_fold_csv(fold_rows,
              os.path.join(folds_dir, f"{stem}_observational_folds.csv"))

        norms_avg, fold_count = _load_norms_avg(targs, 'best-epoch')
        if norms_avg is not None:
            _plot_scatter(norms_avg, fold_count, deg, targs,
                          os.path.join(scatter_dir, f"{stem}_conviction_scatter.pdf"))
        else:
            print("  No best-epoch norms found — scatter skipped.")

    print(f"\nDone. Heatmaps in: {out_dir}")
    print(f"      Scatters in: {scatter_dir}")
    print(f"      Per-fold CSVs in: {folds_dir}")

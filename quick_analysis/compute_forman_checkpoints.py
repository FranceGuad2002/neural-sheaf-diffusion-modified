#!/usr/bin/env python3
"""
Compute Forman Profile evolution across training checkpoints.
Script counterpart of quick_analysis/fmg-Forman-checkpoints.ipynb.

Discovers all combinations that have a checkpoints/ subdirectory under
results/laplacians/*/normalised-true/ and produces one megaplot per combination.

Run from the repo root:
    python quick_analysis/compute_forman_checkpoints.py
"""

import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
LAP_ROOT = Path("results/laplacians")
OUT_DIR  = Path("quick_analysis/metric_images")

# ── constants ──────────────────────────────────────────────────────────────────
MODEL   = "GeneralSheaf"
EXCLUDE = {"synthetic_exp", "synthetic", "texas"}

# Row order in the megaplot — missing checkpoints are silently skipped.
CHECKPOINT_ORDER = ['epoch-0', 'epoch-1', 'epoch-5', 'epoch-15', 'epoch-200', 'best-epoch', 'last']

NUM_NODES = {
    "roman_empire":   22662,
    "minesweeper":    10000,
    "cora":           2708,
    "citeseer":       3327,
    "pubmed":         19717,
    "cornell":        183,
    "wisconsin":      251,
    "amazon_ratings": 24492,
    "questions":      48921,
    "tolokers":       11758,
}

PALETTE = [
    "#E6A817", "#4169E1", "#E8735A", "#5BAD6F",
    "#9B72B0", "#5BB8C9", "#C97D4E", "#C75B8A",
]


# ── discovery ──────────────────────────────────────────────────────────────────
def discover_combinations():
    """
    Scan LAP_ROOT for combinations that have a checkpoints/ subdirectory.
    For each combination records which checkpoint names are available and
    which folds have all layers present for that checkpoint.
    Returns a sorted list of combo dicts.
    """
    # index[key][ckpt_name][fold] = set of layer indices found
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    pattern = f"*/normalised-true/stalk_dim-*/*-layers/*-hidden/*-epochs/checkpoints/*/{MODEL}_*.pt"
    for pt in LAP_ROOT.glob(pattern):
        parts   = pt.relative_to(LAP_ROOT).parts
        dataset = parts[0]
        if dataset in EXCLUDE:
            continue

        stalk_dim = int(parts[2].replace("stalk_dim-", ""))
        layers_n  = int(parts[3].replace("-layers",   ""))
        hidden    = int(parts[4].replace("-hidden",    ""))
        epochs    = int(parts[5].replace("-epochs",    ""))
        ckpt_name = parts[7]   # parts[6] == "checkpoints"

        m_lf = re.search(r"_layer(\d+)_fold(\d+)", pt.stem)
        m_s  = re.search(r"_seed(\d+)",             pt.stem)
        if not (m_lf and m_s):
            continue

        key = (dataset, stalk_dim, layers_n, hidden, epochs, int(m_s.group(1)))
        index[key][ckpt_name][int(m_lf.group(2))].add(int(m_lf.group(1)))

    combos = []
    for key, ckpt_fold_layers in index.items():
        dataset, stalk_dim, layers_n, hidden, epochs, seed = key
        if dataset not in NUM_NODES:
            print(f"  skip {dataset}: not in NUM_NODES")
            continue

        expected   = set(range(layers_n))
        ckpt_folds = {}

        for ckpt_name, fold_layers in ckpt_fold_layers.items():
            complete = sorted(f for f, lrs in fold_layers.items() if expected <= lrs)
            if complete:
                ckpt_folds[ckpt_name] = complete

        if not ckpt_folds:
            print(f"  skip {key}: no complete fold in any checkpoint")
            continue

        # Discover "last" from top-level .pt files (same as compute_forman_metrics.py).
        top_dir    = (LAP_ROOT / dataset / "normalised-true" /
                      f"stalk_dim-{stalk_dim}" / f"{layers_n}-layers" /
                      f"{hidden}-hidden"        / f"{epochs}-epochs")
        last_index = defaultdict(set)
        for pt in top_dir.glob(f"{MODEL}_{dataset}_layer*_fold*_seed{seed}.pt"):
            m = re.search(r"_layer(\d+)_fold(\d+)", pt.stem)
            if m:
                last_index[int(m.group(2))].add(int(m.group(1)))
        last_complete = sorted(f for f, lrs in last_index.items() if expected <= lrs)
        if last_complete:
            ckpt_folds['last'] = last_complete

        # Load best_epochs.json for accuracy annotations (new dict format).
        best_epochs_data = {}
        bej = top_dir / "best_epochs.json"
        if bej.exists():
            with open(bej) as f:
                best_epochs_data = json.load(f)

        combos.append({
            "dataset":          dataset,
            "stalk_dim":        stalk_dim,
            "layers":           layers_n,
            "hidden":           hidden,
            "epochs":           epochs,
            "seed":             seed,
            "n":                NUM_NODES[dataset],
            "ckpt_folds":       ckpt_folds,
            "top_dir":          top_dir,
            "best_epochs_data": best_epochs_data,
        })

    return sorted(combos, key=lambda c: (c["dataset"], c["stalk_dim"]))


# ── path helper ────────────────────────────────────────────────────────────────
def _lap_path(combo, layer, fold, ckpt_name):
    filename = f"{MODEL}_{combo['dataset']}_layer{layer}_fold{fold}_seed{combo['seed']}.pt"
    if ckpt_name == 'last':
        return combo['top_dir'] / filename
    return combo['top_dir'] / 'checkpoints' / ckpt_name / filename


# ── core maths ─────────────────────────────────────────────────────────────────
def compute_F_diags_from_coo(lap, n, d):
    row_idx = lap[0].long().numpy().astype(np.int64)
    col_idx = lap[1].long().numpy().astype(np.int64)
    values  = lap[2].numpy()

    block_row, block_col = row_idx // d, col_idx // d
    local_r,   local_c   = row_idx  % d, col_idx  % d
    is_diag = (block_row == block_col)

    diag_blocks = np.zeros((n, d, d))
    dm = is_diag
    np.add.at(diag_blocks, (block_row[dm], local_r[dm], local_c[dm]), values[dm])

    off       = ~is_diag
    block_ids = block_row[off] * n + block_col[off]
    unique_ids, inv = np.unique(block_ids, return_inverse=True)
    all_blocks = np.zeros((len(unique_ids), d, d))
    np.add.at(all_blocks, (inv, local_r[off], local_c[off]), values[off])

    nz = np.any(all_blocks != 0, axis=(1, 2))
    abs_blocks = np.zeros_like(all_blocks)
    if np.any(nz):
        U, S, _ = np.linalg.svd(all_blocks[nz])
        abs_blocks[nz] = np.einsum("bij,bj,bkj->bik", U, S, U)

    B_diag = np.zeros((n, d, d))
    np.add.at(B_diag, (unique_ids // n).astype(np.int64), abs_blocks)
    return diag_blocks - B_diag


def build_f0_top(path, n, d):
    lap     = torch.load(path, weights_only=True)
    row_idx = lap[0].long().numpy()
    col_idx = lap[1].long().numpy()

    edges = set()
    for r, c in zip(row_idx, col_idx):
        u, v = int(r) // d, int(c) // d
        if u != v:
            edges.add((min(u, v), max(u, v)))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(sorted(edges))

    L_un       = nx.laplacian_matrix(G, nodelist=list(range(n))).astype(float)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.array(L_un.diagonal()) + 1))
    L_norm     = D_inv_sqrt @ L_un @ D_inv_sqrt

    diag_vals    = np.array(L_norm.diagonal())
    abs_row_sums = np.array(np.abs(L_norm).sum(axis=1)).ravel()
    return 2 * diag_vals - abs_row_sums


# ── row label ──────────────────────────────────────────────────────────────────
def _ckpt_acc_stats(ckpt_name, best_epochs_data):
    """Return (mean_test_acc, std_test_acc) across folds, or (None, None)."""
    accs = []
    for fold_entry in best_epochs_data.values():
        if not isinstance(fold_entry, dict):
            return None, None   # old format — no accuracy data
        entry = fold_entry.get('checkpoints', {}).get(ckpt_name, {})
        acc   = entry.get('test_acc')
        if acc is not None:
            accs.append(float(acc))
    if not accs:
        return None, None
    return float(np.mean(accs)), float(np.std(accs))


def _row_label(ckpt_name, best_epochs_data):
    avg_acc, _ = _ckpt_acc_stats(ckpt_name, best_epochs_data)
    acc_str    = f"  test: {avg_acc*100:.1f}%" if avg_acc is not None else ""

    if ckpt_name == 'best-epoch' and best_epochs_data:
        epoch_vals = []
        for fe in best_epochs_data.values():
            if isinstance(fe, dict):
                epoch_vals.append(fe.get('best_epoch', float('nan')))
            else:
                epoch_vals.append(float(fe))
        avg_best = np.nanmean(epoch_vals) if epoch_vals else float('nan')
        return f"best-epoch\n(avg ≈ {avg_best:.0f}{acc_str})"

    if ckpt_name == 'last':
        epoch_vals = []
        for fe in best_epochs_data.values():
            if isinstance(fe, dict):
                ep = fe.get('checkpoints', {}).get('last', {}).get('epoch')
                if ep is not None:
                    epoch_vals.append(float(ep) + 1)  # 0-based → 1-based
        if epoch_vals:
            return f"last-epoch\n(avg ≈ {np.nanmean(epoch_vals):.0f}{acc_str})"
        return f"last-epoch{chr(10) + acc_str.strip() if acc_str else ''}"

    return f"{ckpt_name}{chr(10) + acc_str.strip() if acc_str else ''}"


# ── per-combination pipeline ───────────────────────────────────────────────────
def run_combination(combo):
    dataset   = combo["dataset"]
    d, layers = combo["stalk_dim"], combo["layers"]
    n         = combo["n"]
    best_epochs_data = combo["best_epochs_data"]
    print(f"\n→ {dataset}  d={d}  layers={layers}")

    # Build topological reference from the first available Laplacian.
    ref_ckpt = next(c for c in CHECKPOINT_ORDER if c in combo["ckpt_folds"])
    ref_fold = combo["ckpt_folds"][ref_ckpt][0]
    f0_top   = build_f0_top(_lap_path(combo, 0, ref_fold, ref_ckpt), n, d)
    f0_norm  = (f0_top - f0_top.mean()) / f0_top.std()

    # Compute metrics for every available checkpoint.
    results = {}
    for ckpt_name in CHECKPOINT_ORDER:
        folds = combo["ckpt_folds"].get(ckpt_name)
        if folds is None:
            continue

        nf  = len(folds)
        buf = np.zeros((nf, layers, n, d))
        for fi, fold in enumerate(folds):
            for layer in range(layers):
                lap = torch.load(_lap_path(combo, layer, fold, ckpt_name),
                                 weights_only=True)
                buf[fi, layer] = np.linalg.eigvalsh(
                    compute_F_diags_from_coo(lap, n, d))
            print(f"   {ckpt_name:12s}  fold {fold} done")

        sp_v = np.zeros((nf, layers, d))
        pe_v = np.zeros((nf, layers, d))
        wa_v = np.zeros((nf, layers, d))

        for fi in range(nf):
            for layer in range(layers):
                for k in range(d):
                    c      = buf[fi, layer, :, k]
                    c_norm = (c - c.mean()) / (c.std() + 1e-9)
                    sp_v[fi, layer, k] = spearmanr(c, f0_top).statistic
                    pe_v[fi, layer, k] = pearsonr(c, f0_top).statistic
                    wa_v[fi, layer, k] = wasserstein_distance(c_norm, f0_norm)

        results[ckpt_name] = {
            'spearman_mean':    sp_v.mean(axis=0),
            'spearman_std':     sp_v.std(axis=0),
            'pearson_mean':     pe_v.mean(axis=0),
            'pearson_std':      pe_v.std(axis=0),
            'wasserstein_mean': wa_v.mean(axis=0),
            'wasserstein_std':  wa_v.std(axis=0),
            'n_folds':          nf,
        }

    if not results:
        print("   no valid checkpoints — skipping plot")
        return

    # Megaplot: rows = checkpoints, cols = Spearman / Pearson / Wasserstein.
    valid       = [c for c in CHECKPOINT_ORDER if c in results]
    colors      = PALETTE[:d]
    layer_ticks = np.arange(1, layers + 1)
    n_rows, n_cols = len(valid), 3

    metric_specs = [
        ('spearman_mean',    'spearman_std',    'Spearman ρ',  (-1, 1)),
        ('pearson_mean',     'pearson_std',     'Pearson r',   (-1, 1)),
        ('wasserstein_mean', 'wasserstein_std', 'Wasserstein', (-0.05, 0.9)),
    ]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for row, ckpt_name in enumerate(valid):
        res       = results[ckpt_name]
        n_avail   = res['n_folds']
        row_label = _row_label(ckpt_name, best_epochs_data)

        for col, (mean_key, std_key, title, ylim) in enumerate(metric_specs):
            ax     = axes[row, col]
            mu_mat = res[mean_key]
            sg_mat = res[std_key]

            for k in range(d):
                mu, sig = mu_mat[:, k], sg_mat[:, k]
                ax.plot(layer_ticks, mu, marker='o', color=colors[k],
                        label=f"eig {k+1}", lw=2)
                ax.fill_between(layer_ticks, mu - sig, mu + sig,
                                alpha=0.2, color=colors[k])

            ax.set_xticks(layer_ticks)
            ax.set_ylim(ylim)
            ax.grid(alpha=0.3)

            if row == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')
            if row == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    f"{row_label}\n({n_avail} fold{'s' if n_avail > 1 else ''})",
                    fontsize=10)
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=8, loc='best')

    fig.suptitle(
        f"Forman Profile Evolution — {dataset}  ({MODEL}, d={d}, {layers} layers)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{dataset}_{MODEL}_d{d}_{layers}L_Norm_True_checkpoints_evolution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations()
    print(f"\nFound {len(combos)} combination(s) with checkpoints:")
    for c in combos:
        print(f"  {c['dataset']:20s}  d={c['stalk_dim']}  layers={c['layers']}  "
              f"checkpoints: {sorted(c['ckpt_folds'])}")

    for combo in combos:
        run_combination(combo)

    print("\nAll done.")

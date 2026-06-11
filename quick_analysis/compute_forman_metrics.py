#!/usr/bin/env python3
"""
Compute Forman Profile metrics for all valid GeneralSheaf experiments found under
results/laplacians/*/normalised-true/, save combined figures to quick_analysis/metric_images/.

Run from the repo root:
    python quick_analysis/compute_forman_metrics.py
"""

import re
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
EXCLUDE = {"synthetic_exp", "synthetic", "texas"}
MODEL   = "GeneralSheaf"

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
    "#E6A817",  # gold
    "#4169E1",  # royal blue
    "#E8735A",  # salmon
    "#5BAD6F",  # sage green
    "#9B72B0",  # soft purple
    "#5BB8C9",  # steel blue
    "#C97D4E",  # terracotta
    "#C75B8A",  # rose
]


# ── discovery ──────────────────────────────────────────────────────────────────
def discover_combinations():
    """
    Scan LAP_ROOT for GeneralSheaf .pt files under normalised-true.
    Returns a sorted list of combo dicts, each with complete fold lists.
    Partial folds (e.g. questions with only 5/10) are kept with however
    many folds are fully complete across all layers.
    """
    index = defaultdict(lambda: defaultdict(set))  # key -> fold -> {layers found}

    pattern = f"*/normalised-true/stalk_dim-*/*-layers/*-hidden/*-epochs/{MODEL}_*.pt"
    for pt in LAP_ROOT.glob(pattern):
        parts   = pt.relative_to(LAP_ROOT).parts
        dataset = parts[0]
        if dataset in EXCLUDE:
            continue

        stalk_dim = int(parts[2].replace("stalk_dim-", ""))
        layers_n  = int(parts[3].replace("-layers", ""))
        hidden    = int(parts[4].replace("-hidden", ""))
        epochs    = int(parts[5].replace("-epochs", ""))

        m_lf = re.search(r"_layer(\d+)_fold(\d+)", pt.stem)
        m_s  = re.search(r"_seed(\d+)",             pt.stem)
        if not (m_lf and m_s):
            continue

        key = (dataset, stalk_dim, layers_n, hidden, epochs, int(m_s.group(1)))
        index[key][int(m_lf.group(2))].add(int(m_lf.group(1)))

    combos = []
    for key, fold_to_layers in index.items():
        dataset, stalk_dim, layers_n, hidden, epochs, seed = key
        if dataset not in NUM_NODES:
            print(f"  skip {dataset}: not in NUM_NODES")
            continue
        expected       = set(range(layers_n))
        complete_folds = sorted(f for f, lrs in fold_to_layers.items()
                                if expected <= lrs)
        if not complete_folds:
            print(f"  skip {key}: no complete fold found")
            continue
        combos.append({
            "dataset":   dataset,
            "stalk_dim": stalk_dim,
            "layers":    layers_n,
            "hidden":    hidden,
            "epochs":    epochs,
            "seed":      seed,
            "folds":     complete_folds,
            "n":         NUM_NODES[dataset],
        })

    return sorted(combos, key=lambda c: (c["dataset"], c["stalk_dim"]))


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

    off        = ~is_diag
    block_ids  = block_row[off] * n + block_col[off]
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
    """Load one laplacian, extract graph, return topological Forman curvature f0_top."""
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


# ── per-combination pipeline ───────────────────────────────────────────────────
def _lap_path(combo, layer, fold):
    c = combo
    return (LAP_ROOT / c["dataset"] / "normalised-true" /
            f"stalk_dim-{c['stalk_dim']}" / f"{c['layers']}-layers" /
            f"{c['hidden']}-hidden"        / f"{c['epochs']}-epochs" /
            f"{MODEL}_{c['dataset']}_layer{layer}_fold{fold}_seed{c['seed']}.pt")


def run_combination(combo):
    dataset   = combo["dataset"]
    d, layers = combo["stalk_dim"], combo["layers"]
    n, folds  = combo["n"], combo["folds"]
    nf        = len(folds)
    print(f"\n→ {dataset}  d={d}  layers={layers}  folds={folds}")

    f0_top = build_f0_top(_lap_path(combo, 0, folds[0]), n, d)

    all_curv_eigs = np.zeros((nf, layers, n, d))
    for fi, fold in enumerate(folds):
        for layer in range(layers):
            lap = torch.load(_lap_path(combo, layer, fold), weights_only=True)
            all_curv_eigs[fi, layer] = np.linalg.eigvalsh(
                compute_F_diags_from_coo(lap, n, d))
        print(f"   fold {fold} done")

    f0_norm = (f0_top - f0_top.mean()) / f0_top.std()
    sp_v  = np.zeros((nf, layers, d))
    pe_v  = np.zeros((nf, layers, d))
    wa_v  = np.zeros((nf, layers, d))
    l2n_v = np.zeros((nf, layers, d))
    l2r_v = np.zeros((nf, layers, d))

    for fi in range(nf):
        for layer in range(layers):
            for k in range(d):
                c      = all_curv_eigs[fi, layer, :, k]
                c_norm = (c - c.mean()) / (c.std() + 1e-9)
                sp_v[fi, layer, k]  = spearmanr(c, f0_top).statistic
                pe_v[fi, layer, k]  = pearsonr(c, f0_top).statistic
                wa_v[fi, layer, k]  = wasserstein_distance(c_norm, f0_norm)
                l2n_v[fi, layer, k] = np.linalg.norm(c_norm - f0_norm)
                l2r_v[fi, layer, k] = np.linalg.norm(c - f0_top)

    def ms(v):
        return v.mean(axis=0), v.std(axis=0)

    sp_mu,  sp_s  = ms(sp_v)
    pe_mu,  pe_s  = ms(pe_v)
    wa_mu,  wa_s  = ms(wa_v)
    l2n_mu, l2n_s = ms(l2n_v)
    l2r_mu, l2r_s = ms(l2r_v)

    colors      = PALETTE[:d]
    layer_ticks = np.arange(1, layers + 1)

    fig = plt.figure(figsize=(15, 8))
    gs  = fig.add_gridspec(2, 6, hspace=0.4, wspace=0.3)
    panels = [
        (fig.add_subplot(gs[0, 0:2]), "Spearman ρ", sp_mu,  sp_s,  (-0.25, 1)),
        (fig.add_subplot(gs[0, 2:4]), "Pearson r",  pe_mu,  pe_s,  (-0.25, 1)),
        (fig.add_subplot(gs[0, 4:6]), "Wasserstein",wa_mu,  wa_s,  (-0.05, 0.85)),
        (fig.add_subplot(gs[1, 1:3]), "L2 raw",     l2r_mu, l2r_s, None),
        (fig.add_subplot(gs[1, 3:5]), "L2 norm",    l2n_mu, l2n_s, None),
    ]
    for i, (ax, title, mu_mat, sig_mat, ylim) in enumerate(panels):
        for k in range(d):
            mu, sig = mu_mat[:, k], sig_mat[:, k]
            ax.plot(layer_ticks, mu, marker="o", color=colors[k],
                    label=f"eig {k+1}", lw=2)
            ax.fill_between(layer_ticks, mu - sig, mu + sig,
                            alpha=0.2, color=colors[k])
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Layer", fontsize=18)
        ax.set_xticks(layer_ticks)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if ylim is not None:
            ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel("Metric value", fontsize=18)
        if i == 3:
            ax.set_ylabel("L2 distance", fontsize=18)

    fold_note = f"  ({nf} folds)" if nf < 10 else ""
    fig.suptitle(
        f"{dataset}  ({MODEL}, d={d}, {layers} layers){fold_note}", fontsize=22)
    out = OUT_DIR / f"{dataset}_{MODEL}_d{d}_{layers}L_Norm_True_all_metrics.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations()
    print(f"\nFound {len(combos)} combinations:")
    for c in combos:
        print(f"  {c['dataset']:20s}  d={c['stalk_dim']}  "
              f"layers={c['layers']}  folds={c['folds']}")

    for combo in combos:
        run_combination(combo)

    print("\nAll done.")

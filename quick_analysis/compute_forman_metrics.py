#!/usr/bin/env python3
"""
Compute Forman Profile metrics for experiments found under results/forman_eigs/.

Run from the repo root:
    python quick_analysis/compute_forman_metrics.py
    python quick_analysis/compute_forman_metrics.py --normalised false
    python quick_analysis/compute_forman_metrics.py --model JointSheafParamsAlt --map_type general
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
FORMAN_ROOT = Path("results/forman_eigs")
OUT_DIR     = Path("quick_analysis/metric_images")

# ── constants ──────────────────────────────────────────────────────────────────
EXCLUDE = {"synthetic_exp", "synthetic", "texas"}

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


JOINT_MODELS = {'JointSheafParams', 'JointSheafParamsAlt'}


def _make_map_tag(model, map_type):
    """Filename suffix; empty for non-Joint models."""
    if model not in JOINT_MODELS:
        return ""
    return f"_first_maps_{map_type}"


# ── args ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--normalised', default='true', choices=['true', 'false'],
                   help='Which normalised-* folder to scan (default: true)')
    p.add_argument('--model',      default='GeneralSheaf',
                   help='Model name (default: GeneralSheaf)')
    p.add_argument('--map_type',   default='identity',
                   choices=['identity', 'diag', 'bundle', 'general'],
                   help='First-map type for Joint models (default: identity)')
    return p.parse_args()


# ── discovery ──────────────────────────────────────────────────────────────────
def discover_combinations(model, normalised, map_tag):
    """
    Scan FORMAN_ROOT for .npy eigenvalue files under {model}/normalised-{normalised}.
    Returns a sorted list of combo dicts, each with complete fold lists.
    """
    index = defaultdict(lambda: defaultdict(set))

    pattern = f"*/{model}{map_tag}/normalised-{normalised}/stalk_dim-*/*-layers/*-hidden/*-epochs/{model}_*.npy"
    for pt in FORMAN_ROOT.glob(pattern):
        parts   = pt.relative_to(FORMAN_ROOT).parts
        dataset = parts[0]
        if dataset in EXCLUDE:
            continue

        stalk_dim = int(parts[3].replace("stalk_dim-", ""))
        layers_n  = int(parts[4].replace("-layers", ""))
        hidden    = int(parts[5].replace("-hidden", ""))
        epochs    = int(parts[6].replace("-epochs", ""))

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
            "dataset":    dataset,
            "stalk_dim":  stalk_dim,
            "layers":     layers_n,
            "hidden":     hidden,
            "epochs":     epochs,
            "seed":       seed,
            "folds":      complete_folds,
            "n":          NUM_NODES[dataset],
            "model":      model,
            "normalised": normalised,
            "map_tag":    map_tag,
        })

    return sorted(combos, key=lambda c: (c["dataset"], c["stalk_dim"]))


# ── per-combination pipeline ───────────────────────────────────────────────────
def _base_dir(combo):
    c = combo
    return (FORMAN_ROOT / c["dataset"] / (c["model"] + c["map_tag"]) / f"normalised-{c['normalised']}" /
            f"stalk_dim-{c['stalk_dim']}" / f"{c['layers']}-layers" /
            f"{c['hidden']}-hidden"        / f"{c['epochs']}-epochs")


def _eigs_path(combo, layer, fold):
    c = combo
    fname = (f"{c['model']}_{c['dataset']}_layer{layer}_fold{fold}"
             f"{c['map_tag']}_seed{c['seed']}.npy")
    return _base_dir(c) / fname


def _f0_top_path(combo, fold):
    c = combo
    return _base_dir(c) / f"f0_top_{c['dataset']}_n{c['n']}_fold{fold}_seed{c['seed']}.npy"


def run_combination(combo):
    dataset    = combo["dataset"]
    d, layers  = combo["stalk_dim"], combo["layers"]
    n, folds   = combo["n"], combo["folds"]
    model      = combo["model"]
    normalised = combo["normalised"]
    nf         = len(folds)
    print(f"\n→ {dataset}  d={d}  layers={layers}  folds={folds}  normalised={normalised}")

    f0_top = np.load(_f0_top_path(combo, folds[0]))

    all_curv_eigs = np.zeros((nf, layers, n, d))
    for fi, fold in enumerate(folds):
        for layer in range(layers):
            all_curv_eigs[fi, layer] = np.load(_eigs_path(combo, layer, fold))
        print(f"   fold {fold} done")

    f0_norm = (f0_top - f0_top.mean()) / f0_top.std()
    sp_v    = np.zeros((nf, layers, d))
    pe_v    = np.zeros((nf, layers, d))
    wa_v    = np.zeros((nf, layers, d))
    rmse_v  = np.zeros((nf, layers, d))
    nrmse_v = np.zeros((nf, layers, d))

    for fi in range(nf):
        for layer in range(layers):
            for k in range(d):
                c      = all_curv_eigs[fi, layer, :, k]
                c_norm = (c - c.mean()) / (c.std() + 1e-9)
                sp_v[fi, layer, k]    = spearmanr(c, f0_top).statistic
                pe_v[fi, layer, k]    = pearsonr(c, f0_top).statistic
                wa_v[fi, layer, k]    = wasserstein_distance(c_norm, f0_norm)
                rmse_v[fi, layer, k]  = np.sqrt(np.mean((c - f0_top) ** 2))
                nrmse_v[fi, layer, k] = np.sqrt(np.mean((c_norm - f0_norm) ** 2))

    def ms(v):
        return v.mean(axis=0), v.std(axis=0)

    sp_mu,    sp_s    = ms(sp_v)
    pe_mu,    pe_s    = ms(pe_v)
    wa_mu,    wa_s    = ms(wa_v)
    rmse_mu,  rmse_s  = ms(rmse_v)
    nrmse_mu, nrmse_s = ms(nrmse_v)

    colors      = PALETTE[:d]
    layer_ticks = np.arange(1, layers + 1)
    norm_cap    = normalised.capitalize()

    fig = plt.figure(figsize=(15, 8))
    gs  = fig.add_gridspec(2, 6, hspace=0.4, wspace=0.3)
    panels = [
        (fig.add_subplot(gs[0, 0:2]), "Spearman ρ",  sp_mu,    sp_s,    (-0.25, 1)),
        (fig.add_subplot(gs[0, 2:4]), "Pearson r",   pe_mu,    pe_s,    (-0.25, 1)),
        (fig.add_subplot(gs[0, 4:6]), "Wasserstein", wa_mu,    wa_s,    (-0.05, 0.85)),
        (fig.add_subplot(gs[1, 1:3]), "RMSE",        rmse_mu,  rmse_s,  (-0.05, 0.6)),
        (fig.add_subplot(gs[1, 3:5]), "NRMSE",       nrmse_mu, nrmse_s, (-0.05, 1.6)),
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
            ax.set_ylabel("Distance", fontsize=18)

    fold_note = f"  ({nf} folds)" if nf < 10 else ""
    fig.suptitle(
        f"{dataset}  ({model}, d={d}, {layers} layers, norm={norm_cap}){fold_note}",
        fontsize=22)
    out = OUT_DIR / f"{dataset}_{model}_d{d}_{layers}L_Norm_{norm_cap}_all_metrics.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args       = _parse_args()
    model      = args.model
    normalised = args.normalised
    map_tag    = _make_map_tag(model, args.map_type)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations(model, normalised, map_tag)
    print(f"\nFound {len(combos)} combinations (model={model}, normalised={normalised}, map_tag={map_tag!r}):")
    for c in combos:
        print(f"  {c['dataset']:20s}  d={c['stalk_dim']}  "
              f"layers={c['layers']}  folds={c['folds']}")

    for combo in combos:
        run_combination(combo)

    print("\nAll done.")

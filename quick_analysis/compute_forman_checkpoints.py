#!/usr/bin/env python3
"""
Compute Forman Profile evolution across training checkpoints.

Discovers all combinations that have a checkpoints/ subdirectory under
results/forman_eigs/*/normalised-{normalised}/ and produces:
  - one megaplot (rows=checkpoints, cols=metrics) per combination
  - one heatmap  (mean metric aggregated over eigenvalues) per combination

Run from the repo root:
    python quick_analysis/compute_forman_checkpoints.py
    python quick_analysis/compute_forman_checkpoints.py --normalised false
    python quick_analysis/compute_forman_checkpoints.py --model JointSheafParamsAlt --learn_first_maps true
"""

import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
FORMAN_ROOT = Path("results/forman_eigs")
OUT_DIR     = Path("quick_analysis/checkpoint_forman")

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


# ── args ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--normalised',       default='true',  choices=['true', 'false'],
                   help='Which normalised-* folder to scan (default: true)')
    p.add_argument('--model',            default='GeneralSheaf',
                   help='Model name (default: GeneralSheaf)')
    p.add_argument('--learn_first_maps', default='false', choices=['true', 'false'],
                   help='For JointSheafParamsAlt: learn_first_maps value (default: false)')
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────
def _ckpt_sort_key(name):
    """Sort epoch-N checkpoints numerically, 'last' always at the end."""
    if name == 'last':
        return float('inf')
    m = re.match(r'epoch-(\d+)', name)
    return int(m.group(1)) if m else float('inf') - 1


# ── discovery ──────────────────────────────────────────────────────────────────
def discover_combinations(model, normalised, lfm_tag):
    """
    Scan FORMAN_ROOT for combinations that have a checkpoints/ subdirectory.
    For each combination records which checkpoint names are available and
    which folds have all layers present for that checkpoint.
    Returns a sorted list of combo dicts.
    """
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    pattern = (f"*/normalised-{normalised}/stalk_dim-*/*-layers/*-hidden/*-epochs"
               f"/checkpoints/*/{model}_*.npy")
    for pt in FORMAN_ROOT.glob(pattern):
        if lfm_tag and lfm_tag not in pt.stem:
            continue
        parts   = pt.relative_to(FORMAN_ROOT).parts
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

        top_dir = (FORMAN_ROOT / dataset / f"normalised-{normalised}" /
                   f"stalk_dim-{stalk_dim}" / f"{layers_n}-layers" /
                   f"{hidden}-hidden"        / f"{epochs}-epochs")

        # Discover "last" from top-level .npy files.
        last_index = defaultdict(set)
        for pt in top_dir.glob(f"{model}_{dataset}_layer*_fold*_seed{seed}.npy"):
            if lfm_tag and lfm_tag not in pt.stem:
                continue
            m = re.search(r"_layer(\d+)_fold(\d+)", pt.stem)
            if m:
                last_index[int(m.group(2))].add(int(m.group(1)))
        last_complete = sorted(f for f, lrs in last_index.items() if expected <= lrs)
        if last_complete:
            ckpt_folds['last'] = last_complete

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
            "model":            model,
            "normalised":       normalised,
            "lfm_tag":          lfm_tag,
            "ckpt_folds":       ckpt_folds,
            "top_dir":          top_dir,
            "best_epochs_data": best_epochs_data,
        })

    return sorted(combos, key=lambda c: (c["dataset"], c["stalk_dim"]))


# ── path helpers ───────────────────────────────────────────────────────────────
def _eigs_path(combo, layer, fold, ckpt_name):
    c     = combo
    fname = (f"{c['model']}_{c['dataset']}_layer{layer}_fold{fold}"
             f"{c['lfm_tag']}_seed{c['seed']}.npy")
    if ckpt_name == 'last':
        return c['top_dir'] / fname
    return c['top_dir'] / 'checkpoints' / ckpt_name / fname


def _f0_top_path(combo, fold):
    c = combo
    return c['top_dir'] / f"f0_top_{c['dataset']}_n{c['n']}_fold{fold}_seed{c['seed']}.npy"


# ── row label helpers ──────────────────────────────────────────────────────────
def _ckpt_acc_stats(ckpt_name, best_epochs_data):
    accs = []
    for fold_entry in best_epochs_data.values():
        if not isinstance(fold_entry, dict):
            return None, None
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

    if ckpt_name == 'last':
        epoch_vals = []
        for fe in best_epochs_data.values():
            if isinstance(fe, dict):
                ep = fe.get('checkpoints', {}).get('last', {}).get('epoch')
                if ep is not None:
                    epoch_vals.append(float(ep) + 1)
        if epoch_vals:
            return f"last-epoch\n(avg ≈ {np.nanmean(epoch_vals):.0f}{acc_str})"
        return f"last-epoch{chr(10) + acc_str.strip() if acc_str else ''}"

    return f"{ckpt_name}{chr(10) + acc_str.strip() if acc_str else ''}"


def _short_label(ckpt_name, best_epochs_data):
    """Compact single-line label for heatmap y-axis."""
    if ckpt_name == 'last':
        epoch_vals = [fe.get('checkpoints', {}).get('last', {}).get('epoch')
                      for fe in best_epochs_data.values() if isinstance(fe, dict)]
        epoch_vals = [float(e) + 1 for e in epoch_vals if e is not None]
        return f"last(~{np.nanmean(epoch_vals):.0f})" if epoch_vals else "last"
    return ckpt_name


# ── per-combination pipeline ───────────────────────────────────────────────────
def run_combination(combo):
    dataset    = combo["dataset"]
    d, layers  = combo["stalk_dim"], combo["layers"]
    n          = combo["n"]
    model      = combo["model"]
    normalised = combo["normalised"]
    norm_cap   = normalised.capitalize()
    best_epochs_data = combo["best_epochs_data"]
    print(f"\n→ {dataset}  d={d}  layers={layers}  normalised={normalised}")

    ordered  = sorted(combo["ckpt_folds"].keys(), key=_ckpt_sort_key)

    ref_ckpt = ordered[0]
    ref_fold = combo["ckpt_folds"][ref_ckpt][0]
    f0_top   = np.load(_f0_top_path(combo, ref_fold))
    f0_norm  = (f0_top - f0_top.mean()) / f0_top.std()

    results = {}
    for ckpt_name in ordered:
        folds = combo["ckpt_folds"].get(ckpt_name)
        if folds is None:
            continue

        nf  = len(folds)
        buf = np.zeros((nf, layers, n, d))
        for fi, fold in enumerate(folds):
            for layer in range(layers):
                buf[fi, layer] = np.load(_eigs_path(combo, layer, fold, ckpt_name))
            print(f"   {ckpt_name:12s}  fold {fold} done")

        sp_v    = np.zeros((nf, layers, d))
        pe_v    = np.zeros((nf, layers, d))
        wa_v    = np.zeros((nf, layers, d))
        rmse_v  = np.zeros((nf, layers, d))
        nrmse_v = np.zeros((nf, layers, d))

        for fi in range(nf):
            for layer in range(layers):
                for k in range(d):
                    c      = buf[fi, layer, :, k]
                    c_norm = (c - c.mean()) / (c.std() + 1e-9)
                    sp_v[fi, layer, k]    = spearmanr(c, f0_top).statistic
                    pe_v[fi, layer, k]    = pearsonr(c, f0_top).statistic
                    wa_v[fi, layer, k]    = wasserstein_distance(c_norm, f0_norm)
                    rmse_v[fi, layer, k]  = np.sqrt(np.mean((c - f0_top) ** 2))
                    nrmse_v[fi, layer, k] = np.sqrt(np.mean((c_norm - f0_norm) ** 2))

        results[ckpt_name] = {
            'spearman_mean':    sp_v.mean(axis=0),
            'spearman_std':     sp_v.std(axis=0),
            'pearson_mean':     pe_v.mean(axis=0),
            'pearson_std':      pe_v.std(axis=0),
            'wasserstein_mean': wa_v.mean(axis=0),
            'wasserstein_std':  wa_v.std(axis=0),
            'rmse_mean':        rmse_v.mean(axis=0),
            'rmse_std':         rmse_v.std(axis=0),
            'nrmse_mean':       nrmse_v.mean(axis=0),
            'nrmse_std':        nrmse_v.std(axis=0),
            'n_folds':          nf,
        }

    if not results:
        print("   no valid checkpoints — skipping plots")
        return

    # ── megaplot ───────────────────────────────────────────────────────────────
    colors      = PALETTE[:d]
    layer_ticks = np.arange(1, layers + 1)
    n_rows      = len(ordered)
    n_cols      = 5

    metric_specs = [
        ('spearman_mean',    'spearman_std',    'Spearman ρ',  (-1, 1)),
        ('pearson_mean',     'pearson_std',     'Pearson r',   (-1, 1)),
        ('wasserstein_mean', 'wasserstein_std', 'Wasserstein', (-0.05, 0.9)),
        ('rmse_mean',        'rmse_std',        'RMSE',        (-0.05, 0.6)),
        ('nrmse_mean',       'nrmse_std',       'NRMSE',       (-0.05, 1.8)),
    ]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for row, ckpt_name in enumerate(ordered):
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
        f"Forman Profile Evolution — {dataset}  ({model}, d={d}, {layers}L, norm={norm_cap})",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    lfm_tag   = combo["lfm_tag"]          # e.g. "_learn_first_maps-true" or ""
    fname_tag = lfm_tag.lstrip("_")        # drop leading underscore for readability
    fname_tag = f"_{fname_tag}" if fname_tag else ""
    out = OUT_DIR / f"{dataset}_{model}_d{d}_{layers}L_Norm_{norm_cap}{fname_tag}_checkpoints_evolution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   saved megaplot → {out}")

    # ── heatmap ────────────────────────────────────────────────────────────────
    # Layout: GridSpec(2, 6) — mirrors the megaplot convention
    #   row 0: Spearman [0:2] | Pearson [2:4] | Wasserstein [4:6]
    #   row 1 (centred): RMSE [1:3] | NRMSE [3:5]
    #
    # Color convention:
    #   Correlations (RdBu_r): red = positive/aligned, blue = anticorrelated
    #   Distances   (YlOrRd):  red = high/bad,         yellow = low/good
    #
    # Fixed vmax for distance metrics — same reference scale as the megaplot:
    #   Wasserstein 0.9 | RMSE 0.6 | NRMSE 1.8
    n_ckpts = len(ordered)

    # (key, title, cmap, vmin, vmax, diverging)
    heatmap_specs = [
        ('spearman_mean',    'Spearman ρ',  'RdBu_r', -1,  1,   True),
        ('pearson_mean',     'Pearson r',   'RdBu_r', -1,  1,   True),
        ('wasserstein_mean', 'Wasserstein', 'YlOrRd',  0,  0.9, False),
        ('rmse_mean',        'RMSE',        'YlOrRd',  0,  0.6, False),
        ('nrmse_mean',       'NRMSE',       'YlOrRd',  0,  1.8, False),
    ]

    mats = {}
    for key, *_ in heatmap_specs:
        mat = np.full((n_ckpts, layers), np.nan)
        for i, ckpt_name in enumerate(ordered):
            if ckpt_name in results:
                mat[i] = results[ckpt_name][key].mean(axis=1)  # mean over d → (layers,)
        mats[key] = mat

    cell_h  = 0.55
    cell_w  = 0.90
    panel_w = layers * cell_w
    panel_h = n_ckpts * cell_h
    gap     = 3.0
    left_m  = 0.9;  right_m = 0.3
    top_m   = 1.0;  bot_m   = 0.6

    fig_w = left_m + 3 * panel_w + 2 * gap + 3 * 0.18 + right_m
    fig_h = top_m  + 2 * panel_h + 1.0 + bot_m

    fig2 = plt.figure(figsize=(fig_w, fig_h))
    gs   = fig2.add_gridspec(2, 6,
                             left=left_m/fig_w, right=1-right_m/fig_w,
                             top=1-top_m/fig_h, bottom=bot_m/fig_h,
                             hspace=1.0/panel_h,
                             wspace=gap/panel_w)

    ax_sp  = fig2.add_subplot(gs[0, 0:2])
    ax_pe  = fig2.add_subplot(gs[0, 2:4])
    ax_wa  = fig2.add_subplot(gs[0, 4:6])
    ax_rm  = fig2.add_subplot(gs[1, 1:3])
    ax_nr  = fig2.add_subplot(gs[1, 3:5])
    panel_axes_list = [ax_sp, ax_pe, ax_wa, ax_rm, ax_nr]

    y_labels = [_short_label(c, best_epochs_data) for c in ordered]

    for ax, (key, title, cmap, vmin, vmax, diverging) in zip(panel_axes_list, heatmap_specs):
        mat = mats[key]

        if diverging:
            norm_obj = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm_obj,
                           interpolation='nearest')
        else:
            im = ax.imshow(mat, aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation='nearest')

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=6)
        ax.set_xlabel("Layer", fontsize=15)
        ax.set_xticks(range(layers))
        ax.set_xticklabels(layer_ticks, fontsize=15)
        ax.set_yticks(range(n_ckpts))
        ax.set_yticklabels(y_labels, fontsize=10)

        for i in range(n_ckpts):
            for j in range(layers):
                val = mat[i, j]
                if not np.isnan(val):
                    intensity = abs(val) / max(abs(vmin), abs(vmax)) if diverging else val / vmax
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=15,
                            color='white' if intensity > 0.6 else 'black')

    fig2.suptitle(
        f"Forman Alignment Heatmap — {dataset}  ({model}, d={d}, {layers}L, norm={norm_cap})\n"
        f"(values = mean across {d} eigenvalues, averaged over folds)",
        fontsize=20, y=1.0,
    )
    out2 = OUT_DIR / f"{dataset}_{model}_d{d}_{layers}L_Norm_{norm_cap}{fname_tag}_heatmap.pdf"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"   saved heatmap  → {out2}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args       = _parse_args()
    model      = args.model
    normalised = args.normalised
    lfm_tag    = (f"_learn_first_maps-{args.learn_first_maps}"
                  if model == 'JointSheafParamsAlt' else "")

    # Output folder routing:
    #   Joint models (any variant)          → checkpoint_forman_joint/
    #   Bodnar models + normalised=false    → checkpoint_forman_normalised_false/
    #   Bodnar models + normalised=true     → checkpoint_forman/
    JOINT_MODELS = {'JointSheafParamsAlt'}
    if model in JOINT_MODELS:
        OUT_DIR = Path("quick_analysis/checkpoint_forman_joint")
    elif normalised == 'false':
        OUT_DIR = Path("quick_analysis/checkpoint_forman_normalised_false")
    else:
        OUT_DIR = Path("quick_analysis/checkpoint_forman")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations(model, normalised, lfm_tag)
    print(f"\nFound {len(combos)} combination(s) (model={model}, normalised={normalised}):")
    for c in combos:
        print(f"  {c['dataset']:20s}  d={c['stalk_dim']}  layers={c['layers']}  "
              f"checkpoints: {sorted(c['ckpt_folds'])}")

    for combo in combos:
        run_combination(combo)

    print("\nAll done.")

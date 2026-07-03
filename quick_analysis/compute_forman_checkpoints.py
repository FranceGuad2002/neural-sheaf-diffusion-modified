#!/usr/bin/env python3
"""
Compute Forman Profile evolution across training checkpoints.

Discovers all combinations that have a checkpoints/ subdirectory under
results/forman_eigs/*/normalised-{normalised}/ and produces:
  - one megaplot (rows=checkpoints, cols=metrics) per combination
  - one heatmap  (mean metric aggregated over eigenvalues) per combination

Output folder structure:
  quick_analysis/megaplots/<model>/normalised_{true|false}/
  quick_analysis/heatmaps/<model>/normalised_{true|false}/
  For Joint models an extra level: .../first_maps_{map_type}/

Run from the repo root:
    python quick_analysis/compute_forman_checkpoints.py
    python quick_analysis/compute_forman_checkpoints.py --normalised false
    python quick_analysis/compute_forman_checkpoints.py --model JointSheafParamsAlt --map_type general
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


# ── folder routing ─────────────────────────────────────────────────────────────
def _build_out_dirs(model, normalised, map_type, use_other=False):
    """Return (mega_dir, heat_dir) for this run configuration.

    Content normalisation = training normalisation XOR use_other, so the
    folder name always reflects what the eigenvalues actually represent.
    """
    # Reflect the content, not the training flag
    is_norm_content = (normalised == 'true') != use_other
    content_tag = 'normalised_true' if is_norm_content else 'normalised_false'
    if use_other:
        content_tag += '_other'          # marks that maps came from opposite training run
    if model in JOINT_MODELS:
        sub = Path(content_tag) / f"first_maps_{map_type or 'identity'}"
    else:
        sub = Path(content_tag)
    mega_dir = Path("quick_analysis/megaplots") / model / sub
    heat_dir = Path("quick_analysis/heatmaps")  / model / sub
    return mega_dir, heat_dir


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
    p.add_argument('--use_other',  action='store_true', default=False,
                   help='Load eigs from other_Laplacian/ subfolders (opposite normalisation content)')
    p.add_argument('--do_other_comparison', action='store_true', default=False,
                   help='Cross-compare primary norm=true vs other_Laplacian of norm=false (and vice versa)')
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────
def _ckpt_sort_key(name):
    """Sort epoch-N checkpoints numerically, 'last' always at the end."""
    if name == 'last':
        return float('inf')
    m = re.match(r'epoch-(\d+)', name)
    return int(m.group(1)) if m else float('inf') - 1


# ── discovery ──────────────────────────────────────────────────────────────────
def discover_combinations(model, normalised, map_tag, use_other=False):
    """
    Scan FORMAN_ROOT for combinations that have a checkpoints/ subdirectory.
    When use_other=True, looks inside other_Laplacian/ subfolders so eigs
    from the opposite-normalisation Laplacian are loaded instead.
    Returns a sorted list of combo dicts.
    """
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    other_seg    = 'other_Laplacian/' if use_other else ''
    other_offset = 1                  if use_other else 0

    pattern = (f"*/{model}{map_tag}/normalised-{normalised}/stalk_dim-*/*-layers/*-hidden/*-epochs"
               f"/{other_seg}checkpoints/*/{model}_*.npy")
    for pt in FORMAN_ROOT.glob(pattern):
        parts   = pt.relative_to(FORMAN_ROOT).parts
        dataset = parts[0]
        if dataset in EXCLUDE:
            continue

        stalk_dim = int(parts[3].replace("stalk_dim-", ""))
        layers_n  = int(parts[4].replace("-layers",   ""))
        hidden    = int(parts[5].replace("-hidden",    ""))
        epochs    = int(parts[6].replace("-epochs",    ""))
        ckpt_name = parts[8 + other_offset]   # shifted by 1 when other_Laplacian/ is present

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

        top_dir   = (FORMAN_ROOT / dataset / (model + map_tag) / f"normalised-{normalised}" /
                     f"stalk_dim-{stalk_dim}" / f"{layers_n}-layers" /
                     f"{hidden}-hidden"        / f"{epochs}-epochs")
        eigs_root = top_dir / 'other_Laplacian' if use_other else top_dir

        # Discover "last" from top-level .npy files in eigs_root
        last_index = defaultdict(set)
        for pt in eigs_root.glob(f"{model}_{dataset}_layer*_fold*_seed{seed}.npy"):
            m = re.search(r"_layer(\d+)_fold(\d+)", pt.stem)
            if m:
                last_index[int(m.group(2))].add(int(m.group(1)))
        last_complete = sorted(f for f, lrs in last_index.items() if expected <= lrs)
        if last_complete:
            ckpt_folds['last'] = last_complete

        best_epochs_data = {}
        bej = top_dir / "best_epochs.json"   # always in the primary run folder
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
            "map_tag":          map_tag,
            "ckpt_folds":       ckpt_folds,
            "top_dir":          top_dir,    # primary run folder (f0_top lives here)
            "eigs_root":        eigs_root,  # where to load eigenvalue .npy files from
            "best_epochs_data": best_epochs_data,
        })

    return sorted(combos, key=lambda c: (c["dataset"], c["stalk_dim"]))


# ── path helpers ───────────────────────────────────────────────────────────────
def _eigs_path(combo, layer, fold, ckpt_name):
    c     = combo
    fname = (f"{c['model']}_{c['dataset']}_layer{layer}_fold{fold}"
             f"{c['map_tag']}_seed{c['seed']}.npy")
    if ckpt_name == 'last':
        return c['eigs_root'] / fname
    return c['eigs_root'] / 'checkpoints' / ckpt_name / fname


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
def run_combination(combo, mega_dir, heat_dir, is_norm_content):
    dataset    = combo["dataset"]
    d, layers  = combo["stalk_dim"], combo["layers"]
    n          = combo["n"]
    model      = combo["model"]
    normalised = combo["normalised"]
    norm_cap   = normalised.capitalize()
    best_epochs_data = combo["best_epochs_data"]
    print(f"\n→ {dataset}  d={d}  layers={layers}  normalised={normalised}  norm_content={is_norm_content}")

    ordered  = sorted(combo["ckpt_folds"].keys(), key=_ckpt_sort_key)

    ref_fold    = combo["ckpt_folds"][ordered[0]][0]
    f0_top_path = _f0_top_path(combo, ref_fold)
    if not f0_top_path.exists():
        print(f"   skip: f0_top not found ({f0_top_path}) — run predates topological reference saving")
        return
    f0_top  = np.load(f0_top_path)
    f0_norm = (f0_top - f0_top.mean()) / f0_top.std()

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

        if is_norm_content:
            # Normalized content: all 5 metrics vs topological curvature
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
                'spearman_mean':    sp_v.mean(axis=0),    'spearman_std':    sp_v.std(axis=0),
                'pearson_mean':     pe_v.mean(axis=0),    'pearson_std':     pe_v.std(axis=0),
                'wasserstein_mean': wa_v.mean(axis=0),    'wasserstein_std': wa_v.std(axis=0),
                'rmse_mean':        rmse_v.mean(axis=0),  'rmse_std':        rmse_v.std(axis=0),
                'nrmse_mean':       nrmse_v.mean(axis=0), 'nrmse_std':       nrmse_v.std(axis=0),
                'n_folds':          nf,
            }
        else:
            # Unnormalized content: mixed references — top curvature and zero vector
            wa_top_v    = np.zeros((nf, layers, d))
            wa_zero_v   = np.zeros((nf, layers, d))
            pe_v        = np.zeros((nf, layers, d))
            nrmse_v     = np.zeros((nf, layers, d))
            rmse_zero_v = np.zeros((nf, layers, d))
            zero_ref    = np.zeros(n)

            for fi in range(nf):
                for layer in range(layers):
                    for k in range(d):
                        c      = buf[fi, layer, :, k]
                        c_norm = (c - c.mean()) / (c.std() + 1e-9)
                        wa_top_v[fi, layer, k]    = wasserstein_distance(c_norm, f0_norm)
                        wa_zero_v[fi, layer, k]   = wasserstein_distance(c, zero_ref)
                        pe_v[fi, layer, k]        = pearsonr(c, f0_top).statistic
                        nrmse_v[fi, layer, k]     = np.sqrt(np.mean((c_norm - f0_norm) ** 2))
                        rmse_zero_v[fi, layer, k] = np.sqrt(np.mean(c ** 2))

            results[ckpt_name] = {
                'wa_top_mean':     wa_top_v.mean(axis=0),    'wa_top_std':     wa_top_v.std(axis=0),
                'wa_zero_mean':    wa_zero_v.mean(axis=0),   'wa_zero_std':    wa_zero_v.std(axis=0),
                'pearson_mean':    pe_v.mean(axis=0),        'pearson_std':    pe_v.std(axis=0),
                'nrmse_mean':      nrmse_v.mean(axis=0),     'nrmse_std':      nrmse_v.std(axis=0),
                'rmse_zero_mean':  rmse_zero_v.mean(axis=0), 'rmse_zero_std':  rmse_zero_v.std(axis=0),
                'n_folds':         nf,
            }

    if not results:
        print("   no valid checkpoints — skipping plots")
        return

    # ── megaplot ───────────────────────────────────────────────────────────────
    colors      = PALETTE[:d]
    layer_ticks = np.arange(1, layers + 1)
    n_rows      = len(ordered)
    n_cols      = 5

    if is_norm_content:
        metric_specs = [
            ('spearman_mean',    'spearman_std',    'Spearman ρ',   (-1, 1)),
            ('pearson_mean',     'pearson_std',     'Pearson r',    (-1, 1)),
            ('wasserstein_mean', 'wasserstein_std', 'Wasserstein',  (-0.05, 0.9)),
            ('rmse_mean',        'rmse_std',        'RMSE',         (-0.05, 0.6)),
            ('nrmse_mean',       'nrmse_std',       'NRMSE',        (-0.05, 1.8)),
        ]
    else:
        metric_specs = [
            ('wa_top_mean',    'wa_top_std',    'Wass. vs Top', (-0.05, 0.9)),
            ('wa_zero_mean',   'wa_zero_std',   'Wass. vs 0',   (None, None)),
            ('pearson_mean',   'pearson_std',   'Pearson r',    (-1, 1)),
            ('nrmse_mean',     'nrmse_std',     'NRMSE vs Top', (-0.05, 1.8)),
            ('rmse_zero_mean', 'rmse_zero_std', 'RMSE vs 0',    (None, None)),
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
            if ylim[0] is not None or ylim[1] is not None:
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
    map_tag   = combo["map_tag"]
    fname_tag = map_tag.lstrip("_")
    fname_tag = f"_{fname_tag}" if fname_tag else ""
    mega_dir.mkdir(parents=True, exist_ok=True)
    out = mega_dir / f"{dataset}_{model}_d{d}_{layers}L_Norm_{norm_cap}{fname_tag}_evolution.pdf"
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

    # (key, title, cmap, vmin, vmax, diverging)  — vmax=None means auto from data
    if is_norm_content:
        heatmap_specs = [
            ('spearman_mean',    'Spearman ρ',   'RdBu_r', -1,  1,    True),
            ('pearson_mean',     'Pearson r',    'RdBu_r', -1,  1,    True),
            ('wasserstein_mean', 'Wasserstein',  'YlOrRd',  0,  0.9,  False),
            ('rmse_mean',        'RMSE',         'YlOrRd',  0,  0.6,  False),
            ('nrmse_mean',       'NRMSE',        'YlOrRd',  0,  1.8,  False),
        ]
    else:
        heatmap_specs = [
            ('wa_top_mean',    'Wass. vs Top',  'YlOrRd',  0,  0.9,  False),
            ('wa_zero_mean',   'Wass. vs 0',    'YlOrRd',  0,  None, False),
            ('pearson_mean',   'Pearson r',     'RdBu_r', -1,  1,    True),
            ('nrmse_mean',     'NRMSE vs Top',  'YlOrRd',  0,  1.8,  False),
            ('rmse_zero_mean', 'RMSE vs 0',     'YlOrRd',  0,  None, False),
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
        mat  = mats[key]
        # Resolve auto vmax from data when not fixed
        vmax_eff = vmax if vmax is not None else float(np.nanmax(np.abs(mat))) or 1.0

        if diverging:
            norm_obj = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax_eff)
            im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm_obj,
                           interpolation='nearest')
        else:
            im = ax.imshow(mat, aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax_eff, interpolation='nearest')

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
                    intensity = (abs(val) / max(abs(vmin), abs(vmax_eff))
                                 if diverging else val / vmax_eff)
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=15,
                            color='white' if intensity > 0.6 else 'black')

    fig2.suptitle(
        f"Forman Alignment Heatmap — {dataset}  ({model}, d={d}, {layers}L, norm={norm_cap})\n"
        f"(values = mean across {d} eigenvalues, averaged over folds)",
        fontsize=20, y=1.0,
    )
    heat_dir.mkdir(parents=True, exist_ok=True)
    out2 = heat_dir / f"{dataset}_{model}_d{d}_{layers}L_Norm_{norm_cap}{fname_tag}_heatmap.pdf"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"   saved heatmap  → {out2}")


# ── cross-comparison pipeline ──────────────────────────────────────────────────
def discover_comparison_pairs(combos_a, combos_b):
    """
    Match combos from two sources by (dataset, stalk_dim, layers, seed).
    Returns a list of (combo_a, combo_b) tuples for matched configs.
    """
    index_b = {(c['dataset'], c['stalk_dim'], c['layers'], c['seed']): c for c in combos_b}
    pairs   = []
    for ca in combos_a:
        key = (ca['dataset'], ca['stalk_dim'], ca['layers'], ca['seed'])
        if key in index_b:
            pairs.append((ca, index_b[key]))
        else:
            print(f"  no match for {ca['dataset']} d={ca['stalk_dim']} L={ca['layers']}")
    return pairs


def run_comparison(combo_a, combo_b, mega_dir, heat_dir, label_a, label_b):
    """
    Cross-compare Forman eigs from two sources fold-by-fold.
    Metrics (Spearman, Pearson, Wasserstein, RMSE, NRMSE) are computed
    between the two eig distributions per fold / per layer / per checkpoint.
    Produces the same megaplot + heatmap format as run_combination().
    """
    dataset   = combo_a["dataset"]
    d, layers = combo_a["stalk_dim"], combo_a["layers"]
    n         = combo_a["n"]
    model     = combo_a["model"]
    best_epochs_data = combo_a["best_epochs_data"]
    print(f"\n→ COMPARISON {dataset}  d={d}  L={layers}:  {label_a}  vs  {label_b}")

    ordered = sorted(
        set(combo_a["ckpt_folds"]) & set(combo_b["ckpt_folds"]),
        key=_ckpt_sort_key,
    )

    results = {}
    for ckpt_name in ordered:
        folds = sorted(
            set(combo_a["ckpt_folds"].get(ckpt_name, [])) &
            set(combo_b["ckpt_folds"].get(ckpt_name, []))
        )
        if not folds:
            continue

        nf    = len(folds)
        buf_a = np.zeros((nf, layers, n, d))
        buf_b = np.zeros((nf, layers, n, d))
        for fi, fold in enumerate(folds):
            for layer in range(layers):
                buf_a[fi, layer] = np.load(_eigs_path(combo_a, layer, fold, ckpt_name))
                buf_b[fi, layer] = np.load(_eigs_path(combo_b, layer, fold, ckpt_name))
            print(f"   {ckpt_name:12s}  fold {fold} done")

        sp_v    = np.zeros((nf, layers, d))
        pe_v    = np.zeros((nf, layers, d))
        wa_v    = np.zeros((nf, layers, d))
        rmse_v  = np.zeros((nf, layers, d))
        nrmse_v = np.zeros((nf, layers, d))

        for fi in range(nf):
            for layer in range(layers):
                for k in range(d):
                    ca_k    = buf_a[fi, layer, :, k]
                    cb_k    = buf_b[fi, layer, :, k]
                    ca_norm = (ca_k - ca_k.mean()) / (ca_k.std() + 1e-9)
                    cb_norm = (cb_k - cb_k.mean()) / (cb_k.std() + 1e-9)
                    sp_v[fi, layer, k]    = spearmanr(ca_k, cb_k).statistic
                    pe_v[fi, layer, k]    = pearsonr(ca_k, cb_k).statistic
                    wa_v[fi, layer, k]    = wasserstein_distance(ca_norm, cb_norm)
                    rmse_v[fi, layer, k]  = np.sqrt(np.mean((ca_k - cb_k) ** 2))
                    nrmse_v[fi, layer, k] = np.sqrt(np.mean((ca_norm - cb_norm) ** 2))

        results[ckpt_name] = {
            'spearman_mean':    sp_v.mean(axis=0),    'spearman_std':    sp_v.std(axis=0),
            'pearson_mean':     pe_v.mean(axis=0),    'pearson_std':     pe_v.std(axis=0),
            'wasserstein_mean': wa_v.mean(axis=0),    'wasserstein_std': wa_v.std(axis=0),
            'rmse_mean':        rmse_v.mean(axis=0),  'rmse_std':        rmse_v.std(axis=0),
            'nrmse_mean':       nrmse_v.mean(axis=0), 'nrmse_std':       nrmse_v.std(axis=0),
            'n_folds':          nf,
        }

    if not results:
        print("   no matched checkpoints — skipping")
        return

    metric_specs = [
        ('spearman_mean',    'spearman_std',    'Spearman ρ',  (-1, 1)),
        ('pearson_mean',     'pearson_std',     'Pearson r',   (-1, 1)),
        ('wasserstein_mean', 'wasserstein_std', 'Wasserstein', (-0.05, 0.9)),
        ('rmse_mean',        'rmse_std',        'RMSE',        (-0.05, 0.6)),
        ('nrmse_mean',       'nrmse_std',       'NRMSE',       (-0.05, 1.8)),
    ]
    heatmap_specs = [
        ('spearman_mean',    'Spearman ρ',   'RdBu_r', -1,  1,   True),
        ('pearson_mean',     'Pearson r',    'RdBu_r', -1,  1,   True),
        ('wasserstein_mean', 'Wasserstein',  'YlOrRd',  0,  0.9, False),
        ('rmse_mean',        'RMSE',         'YlOrRd',  0,  0.6, False),
        ('nrmse_mean',       'NRMSE',        'YlOrRd',  0,  1.8, False),
    ]

    # ── megaplot ───────────────────────────────────────────────────────────────
    colors      = PALETTE[:d]
    layer_ticks = np.arange(1, layers + 1)
    n_rows, n_cols = len(ordered), 5

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for row, ckpt_name in enumerate(ordered):
        if ckpt_name not in results:
            continue
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
            if ylim[0] is not None or ylim[1] is not None:
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
        f"Forman Cross-Comparison — {dataset}  ({model}, d={d}, {layers}L)\n"
        f"{label_a}  vs  {label_b}",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fname_stem = f"{dataset}_{model}_d{d}_{layers}L_{label_a}_vs_{label_b}"
    mega_dir.mkdir(parents=True, exist_ok=True)
    out = mega_dir / f"{fname_stem}_evolution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   saved megaplot → {out}")

    # ── heatmap ────────────────────────────────────────────────────────────────
    n_ckpts     = len(ordered)
    mats        = {}
    for key, *_ in heatmap_specs:
        mat = np.full((n_ckpts, layers), np.nan)
        for i, ckpt_name in enumerate(ordered):
            if ckpt_name in results:
                mat[i] = results[ckpt_name][key].mean(axis=1)
        mats[key] = mat

    cell_h  = 0.55;  cell_w  = 0.90
    panel_w = layers * cell_w;  panel_h = n_ckpts * cell_h
    gap     = 3.0
    left_m  = 0.9;  right_m = 0.3;  top_m = 1.2;  bot_m = 0.6

    fig_w = left_m + 3 * panel_w + 2 * gap + 3 * 0.18 + right_m
    fig_h = top_m  + 2 * panel_h + 1.0 + bot_m

    fig2 = plt.figure(figsize=(fig_w, fig_h))
    gs   = fig2.add_gridspec(2, 6,
                             left=left_m/fig_w, right=1-right_m/fig_w,
                             top=1-top_m/fig_h, bottom=bot_m/fig_h,
                             hspace=1.0/panel_h, wspace=gap/panel_w)

    panel_axes_list = [
        fig2.add_subplot(gs[0, 0:2]),
        fig2.add_subplot(gs[0, 2:4]),
        fig2.add_subplot(gs[0, 4:6]),
        fig2.add_subplot(gs[1, 1:3]),
        fig2.add_subplot(gs[1, 3:5]),
    ]
    y_labels = [_short_label(c, best_epochs_data) for c in ordered]

    for ax, (key, title, cmap, vmin, vmax, diverging) in zip(panel_axes_list, heatmap_specs):
        mat      = mats[key]
        vmax_eff = vmax if vmax is not None else float(np.nanmax(np.abs(mat))) or 1.0

        if diverging:
            norm_obj = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax_eff)
            im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm_obj,
                           interpolation='nearest')
        else:
            im = ax.imshow(mat, aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax_eff, interpolation='nearest')

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
                    intensity = (abs(val) / max(abs(vmin), abs(vmax_eff))
                                 if diverging else val / vmax_eff)
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=15,
                            color='white' if intensity > 0.6 else 'black')

    fig2.suptitle(
        f"Forman Cross-Comparison — {dataset}  ({model}, d={d}, {layers}L)\n"
        f"{label_a}  vs  {label_b}\n"
        f"(values = mean across {d} eigenvalues, averaged over folds)",
        fontsize=18, y=1.0,
    )
    heat_dir.mkdir(parents=True, exist_ok=True)
    out2 = heat_dir / f"{fname_stem}_heatmap.pdf"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"   saved heatmap  → {out2}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args       = _parse_args()
    model      = args.model
    normalised = args.normalised
    map_tag    = _make_map_tag(model, args.map_type)

    if args.do_other_comparison:
        # Direction A: primary norm=true  vs  other_Laplacian of norm=false
        #   → both represent normalized L, maps trained under different objectives
        combos_norm_true   = discover_combinations(model, 'true',  map_tag, use_other=False)
        combos_other_false = discover_combinations(model, 'false', map_tag, use_other=True)
        pairs_A = discover_comparison_pairs(combos_norm_true, combos_other_false)
        mega_A  = Path('quick_analysis/megaplots/comparison/norm_true_vs_other_norm_false')
        heat_A  = Path('quick_analysis/heatmaps/comparison/norm_true_vs_other_norm_false')
        print(f"\nDirection A — {len(pairs_A)} pair(s): norm_true_primary vs other_norm_false")
        for ca, cb in pairs_A:
            run_comparison(ca, cb, mega_A, heat_A,
                           label_a='norm_true_primary', label_b='other_norm_false')

        # Direction B: primary norm=false  vs  other_Laplacian of norm=true
        #   → both represent unnormalized L, maps trained under different objectives
        combos_norm_false = discover_combinations(model, 'false', map_tag, use_other=False)
        combos_other_true = discover_combinations(model, 'true',  map_tag, use_other=True)
        pairs_B = discover_comparison_pairs(combos_norm_false, combos_other_true)
        mega_B  = Path('quick_analysis/megaplots/comparison/norm_false_vs_other_norm_true')
        heat_B  = Path('quick_analysis/heatmaps/comparison/norm_false_vs_other_norm_true')
        print(f"\nDirection B — {len(pairs_B)} pair(s): norm_false_primary vs other_norm_true")
        for ca, cb in pairs_B:
            run_comparison(ca, cb, mega_B, heat_B,
                           label_a='norm_false_primary', label_b='other_norm_true')

    else:
        use_other           = args.use_other
        mega_dir, heat_dir  = _build_out_dirs(model, normalised, args.map_type, use_other)
        # Content is normalised iff (training was normalised) XOR (reading other_Laplacian)
        is_norm_content     = (normalised == 'true') != use_other

        combos = discover_combinations(model, normalised, map_tag, use_other=use_other)
        print(f"\nFound {len(combos)} combination(s) "
              f"(model={model}, normalised={normalised}, map_tag={map_tag!r}, "
              f"use_other={use_other}, norm_content={is_norm_content}):")
        for c in combos:
            print(f"  {c['dataset']:20s}  d={c['stalk_dim']}  layers={c['layers']}  "
                  f"checkpoints: {sorted(c['ckpt_folds'])}")

        for combo in combos:
            run_combination(combo, mega_dir, heat_dir, is_norm_content)

    print("\nAll done.")

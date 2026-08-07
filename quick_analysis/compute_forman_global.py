#!/usr/bin/env python3
"""
Harvest per-fold, per-layer Forman-profile Wasserstein distances into a single
long-format CSV, for downstream global (across-dataset) aggregation.

This performs no new computation beyond what compute_forman_checkpoints.py
already does per dataset: it reuses its discovery/path helpers and its
per-fold, per-eigenvalue Wasserstein computation, but persists the values
instead of collapsing them straight into a heatmap.

Layers are tagged with a depth bin (first / middle / last) rather than an
absolute layer index, so that datasets trained at different depths (L=2..6)
can be aggregated without rerunning anything at a harmonised layer count.

Run from the repo root (needs the Do_nsd conda env, for pandas):
    python quick_analysis/compute_forman_global.py
    python quick_analysis/compute_forman_global.py --datasets cornell,wisconsin  # quick smoke test
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance

from compute_forman_checkpoints import (
    discover_combinations,
    discover_comparison_pairs,
    _eigs_path,
    _f0_top_path,
    _ckpt_sort_key,
    _make_map_tag,
)

OUT_DIR = Path("quick_analysis/global_metrics/raw")


# ── depth binning ─────────────────────────────────────────────────────────────
def _depth_bin(layer, layers):
    if layer == 0:
        return "first"
    if layer == layers - 1:
        return "last"
    return "middle"


# ── per-(fold, layer) metric ───────────────────────────────────────────────────
def _eigen_avg_wasserstein(eigs, f0_norm, zero_ref):
    """eigs: (n, d) curvature eigenvalues for one (fold, layer, checkpoint).
    Returns (w_vs_top, w_vs_zero), each averaged over the d eigenvalue channels."""
    d = eigs.shape[1]
    w_top  = np.empty(d)
    w_zero = np.empty(d)
    for k in range(d):
        c      = eigs[:, k]
        c_norm = (c - c.mean()) / (c.std() + 1e-9)
        w_top[k]  = wasserstein_distance(c_norm, f0_norm)
        w_zero[k] = wasserstein_distance(c, zero_ref)
    return float(w_top.mean()), float(w_zero.mean())


# ── harvester: profile vs reference(s) ─────────────────────────────────────────
def harvest_vs_reference(model, normalised, map_type, use_other, datasets=None):
    """One row per (dataset, checkpoint, fold, layer, reference)."""
    map_tag = _make_map_tag(model, map_type)
    combos  = discover_combinations(model, normalised, map_tag, use_other=use_other)
    if datasets is not None:
        combos = [c for c in combos if c["dataset"] in datasets]
    content_normalised = (normalised == 'true') != use_other
    rows = []

    for combo in combos:
        dataset, d, layers = combo["dataset"], combo["stalk_dim"], combo["layers"]
        n       = combo["n"]
        ordered = sorted(combo["ckpt_folds"].keys(), key=_ckpt_sort_key)
        if not ordered:
            continue

        ref_fold    = combo["ckpt_folds"][ordered[0]][0]
        f0_top_path = _f0_top_path(combo, ref_fold)
        if not f0_top_path.exists():
            print(f"  skip {dataset} ({model}, norm={normalised}, other={use_other}): no f0_top")
            continue
        f0_top   = np.load(f0_top_path)
        f0_norm  = (f0_top - f0_top.mean()) / f0_top.std()
        zero_ref = np.zeros(n)

        for ckpt_name in ordered:
            folds = combo["ckpt_folds"].get(ckpt_name)
            if not folds:
                continue
            for fold in folds:
                for layer in range(layers):
                    eigs = np.load(_eigs_path(combo, layer, fold, ckpt_name))
                    w_top, w_zero = _eigen_avg_wasserstein(eigs, f0_norm, zero_ref)
                    base = dict(
                        model=model, map_type=map_type or "", normalised=normalised,
                        use_other=use_other, content_normalised=content_normalised,
                        dataset=dataset, stalk_dim=d, layers=layers,
                        checkpoint=ckpt_name, layer=layer,
                        depth_bin=_depth_bin(layer, layers), fold=fold,
                    )
                    rows.append({**base, "reference": "topological", "wasserstein": w_top})
                    rows.append({**base, "reference": "zero",        "wasserstein": w_zero})
        print(f"  {dataset:16s} d={d} L={layers}  "
              f"({model}, norm={normalised}, other={use_other})  done  ({len(rows)} rows so far)")

    return rows


# ── harvester: profile vs profile (cross-comparison) ───────────────────────────
def harvest_cross_comparison(model, direction, normalised_a, use_other_a,
                              normalised_b, use_other_b, datasets=None):
    """One row per (dataset, checkpoint, fold, layer): eigen-averaged Wasserstein
    between two learnt profiles (no fixed reference)."""
    map_tag  = _make_map_tag(model, None)
    combos_a = discover_combinations(model, normalised_a, map_tag, use_other=use_other_a)
    combos_b = discover_combinations(model, normalised_b, map_tag, use_other=use_other_b)
    if datasets is not None:
        combos_a = [c for c in combos_a if c["dataset"] in datasets]
        combos_b = [c for c in combos_b if c["dataset"] in datasets]
    pairs = discover_comparison_pairs(combos_a, combos_b)
    rows  = []

    for combo_a, combo_b in pairs:
        dataset, d, layers = combo_a["dataset"], combo_a["stalk_dim"], combo_a["layers"]
        ordered = sorted(set(combo_a["ckpt_folds"]) & set(combo_b["ckpt_folds"]), key=_ckpt_sort_key)
        for ckpt_name in ordered:
            folds = sorted(set(combo_a["ckpt_folds"].get(ckpt_name, [])) &
                            set(combo_b["ckpt_folds"].get(ckpt_name, [])))
            for fold in folds:
                for layer in range(layers):
                    ea = np.load(_eigs_path(combo_a, layer, fold, ckpt_name))
                    eb = np.load(_eigs_path(combo_b, layer, fold, ckpt_name))
                    w  = np.empty(d)
                    for k in range(d):
                        ca, cb = ea[:, k], eb[:, k]
                        ca_n   = (ca - ca.mean()) / (ca.std() + 1e-9)
                        cb_n   = (cb - cb.mean()) / (cb.std() + 1e-9)
                        w[k]   = wasserstein_distance(ca_n, cb_n)
                    rows.append(dict(
                        model=model, direction=direction,
                        dataset=dataset, stalk_dim=d, layers=layers,
                        checkpoint=ckpt_name, layer=layer,
                        depth_bin=_depth_bin(layer, layers), fold=fold,
                        reference="cross", wasserstein=float(w.mean()),
                    ))
        print(f"  {dataset:16s} d={d} L={layers}  (cross {direction})  done  ({len(rows)} rows so far)")

    return rows


# Table numbering follows the thesis plan: 1-4 main text, 5-8 supplementary.
JOBS_VS_REFERENCE = [
    # (model, normalised, map_type, use_other)
    ("GeneralSheaf",      "true",  None,       False),  # Table 1
    ("GeneralSheaf",      "false", None,       False),  # Table 2
    ("GeneralSheaf",      "false", None,       True),   # Table 5 (other: trained False -> normalised content)
    ("GeneralSheaf",      "true",  None,       True),   # Table 6 (other: trained True  -> unnormalised content)
    ("JointSheafParams",  "false", "identity", False),  # Table 3
    ("JointSheafParams",  "true",  "identity", False),  # Table 4
]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=None,
                    help='Comma-separated dataset filter, for a quick smoke test '
                         '(e.g. --datasets cornell,wisconsin)')
    return p.parse_args()


def main():
    args     = _parse_args()
    datasets = set(args.datasets.split(',')) if args.datasets else None

    all_rows = []
    for model, normalised, map_type, use_other in JOBS_VS_REFERENCE:
        print(f"\n== {model}  norm={normalised}  map_type={map_type}  use_other={use_other} ==")
        all_rows.extend(harvest_vs_reference(model, normalised, map_type, use_other, datasets))

    print("\n== Cross-comparison Direction A (both normalised, GeneralSheaf) ==")
    all_rows.extend(harvest_cross_comparison(
        "GeneralSheaf", "A", "true", False, "false", True, datasets))  # Table 7

    print("\n== Cross-comparison Direction B (both unnormalised, GeneralSheaf) ==")
    all_rows.extend(harvest_cross_comparison(
        "GeneralSheaf", "B", "false", False, "true", True, datasets))  # Table 8

    df  = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{'-'.join(sorted(datasets))}" if datasets else ""
    out = OUT_DIR / f"forman_wasserstein_long{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()

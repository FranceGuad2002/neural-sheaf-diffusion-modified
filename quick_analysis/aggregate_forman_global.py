#!/usr/bin/env python3
"""
Aggregate the long-format Wasserstein table produced by compute_forman_global.py
into per-(config, checkpoint, depth_bin) summary statistics.

Two uncertainty estimates are reported side by side, both computed from the
same raw (dataset, fold, layer) observations:
  - se_across  : SE of the per-dataset means, n = n_datasets. Primary estimate:
                 treats each dataset as one independent unit, which is the
                 right null model for a claim about generalisation across
                 datasets (folds within a dataset are not independent replicates
                 of that claim).
  - se_pooled  : SE pooling every (dataset, fold, layer-in-bin) observation as
                 if independent, n = n_obs. Reported for completeness only;
                 folds within a dataset are correlated, so this understates
                 the true uncertainty and should not be read as a tighter CI.
  - mean       : dataset-equal-weighted mean (mean of per-dataset means), paired
                 with se_across. mean_pooled (obs-weighted) is also kept, for
                 reference; the two are close whenever fold counts per dataset
                 are balanced (10 folds everywhere in this study).

Run from the repo root (needs the Do_nsd conda env, for pandas):
    python quick_analysis/aggregate_forman_global.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("quick_analysis/global_metrics/raw")
OUT_DIR = Path("quick_analysis/global_metrics/agg")

VS_REFERENCE_KEYS = [
    "model", "map_type", "normalised", "use_other", "content_normalised", "reference",
]
CROSS_KEYS = ["model", "direction", "reference"]

CELL_KEYS = ["checkpoint", "depth_bin"]


def _aggregate(df, config_keys):
    group_keys = config_keys + CELL_KEYS

    per_dataset = (
        df.groupby(group_keys + ["dataset"])["wasserstein"]
        .mean()
        .rename("dataset_mean")
        .reset_index()
    )
    across = (
        per_dataset.groupby(group_keys)["dataset_mean"]
        .agg(mean="mean", se_across="sem", n_datasets="size")
        .reset_index()
    )
    pooled = (
        df.groupby(group_keys)["wasserstein"]
        .agg(mean_pooled="mean", se_pooled="sem", n_obs="count")
        .reset_index()
    )
    return across.merge(pooled, on=group_keys, how="left")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("forman_wasserstein_long*.csv"))
    if not raw_files:
        raise SystemExit(f"No raw CSVs found in {RAW_DIR} — run compute_forman_global.py first.")
    df = pd.concat((pd.read_csv(f) for f in raw_files), ignore_index=True)
    print(f"Loaded {len(df)} raw rows from {len(raw_files)} file(s).")

    # GeneralSheaf rows have map_type="" (non-Joint models carry no map_type).
    # An empty CSV field round-trips to NaN, and groupby drops NaN keys by
    # default, which silently dropped every GeneralSheaf row. Restore "".
    df["map_type"] = df["map_type"].fillna("")

    vs_ref = df[df["reference"].isin(["topological", "zero"])].copy()
    cross  = df[df["reference"] == "cross"].copy()

    agg_vs_ref = _aggregate(vs_ref, VS_REFERENCE_KEYS)
    agg_cross  = _aggregate(cross,  CROSS_KEYS)

    out1 = OUT_DIR / "forman_wasserstein_agg_vs_reference.csv"
    out2 = OUT_DIR / "forman_wasserstein_agg_cross.csv"
    agg_vs_ref.to_csv(out1, index=False)
    agg_cross.to_csv(out2, index=False)
    print(f"Wrote {len(agg_vs_ref)} rows -> {out1}")
    print(f"Wrote {len(agg_cross)} rows -> {out2}")


if __name__ == "__main__":
    main()

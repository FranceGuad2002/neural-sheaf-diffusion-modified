#!/usr/bin/env python3
"""
Turn the aggregated Forman-profile Wasserstein tables into LaTeX booktabs
tables, one .tex fragment per table in the thesis plan (Tables 1-8).

Main-text cells show mean +/- se_across (dataset-equal-weighted mean, SE
across datasets). The pooled SE is reported once in the caption as a sanity
check, not per cell (see aggregate_forman_global.py for why).

Cells are shaded with the same YlOrRd colormap used in the per-dataset
heatmap figures (compute_forman_checkpoints.py), vmin=0, vmax = the max
mean Wasserstein value actually shown in that table (or, for tables with
two reference blocks side by side, in that block). Colors are emitted as
named \\definecolor's (deduplicated per table) rather than inline rgb, and
each block gets its own TikZ colorbar to its right.

REQUIRES in your document preamble (not just this fragment):
    \\usepackage[table]{xcolor}
    \\usepackage{tikz}

Run from the repo root (needs the Do_nsd conda env, for pandas/matplotlib):
    python quick_analysis/generate_forman_tables.py
"""

import re
import math
import matplotlib as mpl
import pandas as pd
from pathlib import Path

AGG_DIR = Path("quick_analysis/global_metrics/agg")
OUT_DIR = Path("quick_analysis/tables")

CHECKPOINT_ORDER = ["epoch-0", "epoch-1", "epoch-5", "epoch-15", "epoch-50", "epoch-200", "last"]
CHECKPOINT_LABEL = {
    "epoch-0": "0", "epoch-1": "1", "epoch-5": "5", "epoch-15": "15",
    "epoch-50": "50", "epoch-200": "200", "last": "last",
}
BIN_ORDER = ["first", "middle", "last"]
BIN_LABEL = {"first": "First", "middle": "Middle", "last": "Last"}

CMAP_NAME = "YlOrRd"


# ── color infrastructure ────────────────────────────────────────────────────
class _ColorRegistry:
    """Collects unique (r,g,b) colors used in one table fragment and emits
    \\definecolor lines once, so cells/colorbar reference them by name
    instead of repeating inline rgb literals."""

    def __init__(self, prefix):
        self.prefix = re.sub(r"[^a-zA-Z]", "", prefix) or "clr"
        self._names = {}
        self._defs = []

    def name_for(self, rgb):
        key = tuple(round(c, 4) for c in rgb)
        if key not in self._names:
            name = f"{self.prefix}{len(self._names)}"
            self._names[key] = name
            self._defs.append(
                r"\definecolor{" + name + "}{rgb}{" +
                f"{key[0]},{key[1]},{key[2]}" + "}"
            )
        return self._names[key]

    def definitions(self):
        return "\n".join(self._defs)


def _cmap_rgb(frac, cmap_name=CMAP_NAME):
    r, g, b, _ = mpl.colormaps[cmap_name](max(0.0, min(1.0, frac)))
    return r, g, b


def _cell_colored(row, vmax, registry):
    if row is None or pd.isna(row["mean"]):
        return "--"
    val  = max(0.0, float(row["mean"]))
    frac = val / vmax if vmax > 0 else 0.0
    name = registry.name_for(_cmap_rgb(frac))
    textcolor = "white" if frac > 0.6 else "black"
    return (
        r"\cellcolor{" + name + r"}\textcolor{" + textcolor + "}{$" +
        f"{row['mean']:.3f} \\pm {row['se_across']:.3f}" + "$}"
    )


def _colorbar_tikz(vmax, registry, n_segments=14, height_cm=3.4, width_cm=0.35):
    """Vertical piecewise-constant approximation of the YlOrRd colormap,
    vmin=0 at the bottom, vmax at the top, with a mid tick for reference."""
    if not (vmax > 0) or not math.isfinite(vmax):
        return ""
    seg_h = height_cm / n_segments
    lines = [r"\begin{tikzpicture}"]
    for i in range(n_segments):
        frac = (i + 0.5) / n_segments
        name = registry.name_for(_cmap_rgb(frac))
        y0, y1 = i * seg_h, (i + 1) * seg_h
        lines.append(
            r"\fill[" + name + "] (0," + f"{y0:.4f}" + ") rectangle (" +
            f"{width_cm},{y1:.4f}" + ");"
        )
    lines.append(r"\draw[black,line width=0.3pt] (0,0) rectangle (" +
                 f"{width_cm},{height_cm:.4f}" + ");")
    lines.append(r"\node[right,font=\scriptsize,anchor=west] at (" +
                 f"{width_cm},{height_cm:.4f}" + ") {" + f"{vmax:.2f}" + "};")
    lines.append(r"\node[right,font=\scriptsize,anchor=west] at (" +
                 f"{width_cm},{height_cm / 2:.4f}" + ") {" + f"{vmax / 2:.2f}" + "};")
    lines.append(r"\node[right,font=\scriptsize,anchor=west] at (" +
                 f"{width_cm}" + ",0) {0.00};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _hstack(minipage_blocks):
    """Join several [\\begin{minipage}...\\end{minipage}] line-lists side by
    side with \\hfill glue (each block's last line must be \\end{minipage})."""
    out = []
    for i, block in enumerate(minipage_blocks):
        out.extend(block[:-1])
        if i < len(minipage_blocks) - 1:
            out.append(block[-1] + "%")
            out.append(r"\hfill")
        else:
            out.append(block[-1])
    return out


# ── data lookup ──────────────────────────────────────────────────────────────
def _lookup(df, checkpoint, depth_bin, **filters):
    sub = df
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    sub = sub[(sub["checkpoint"] == checkpoint) & (sub["depth_bin"] == depth_bin)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _collect_means(df, filters):
    vals = []
    for ckpt in CHECKPOINT_ORDER:
        for b in BIN_ORDER:
            row = _lookup(df, ckpt, b, **filters)
            if row is not None and pd.notna(row["mean"]):
                vals.append(float(row["mean"]))
    return vals


def _avg_pooled_se(df, **filters):
    sub = df
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return float("nan")
    return sub["se_pooled"].mean()


def _coverage_note(df, filters, exclusion_note=None):
    """Report the actual n_datasets per bin (data-driven, not assumed to be 10),
    plus an optional free-text reason for known exclusions (e.g. an unstable run)."""
    first_row  = _lookup(df, CHECKPOINT_ORDER[0], "first",  **filters)
    middle_row = _lookup(df, CHECKPOINT_ORDER[0], "middle", **filters)
    n_first  = int(first_row["n_datasets"])  if first_row  is not None else None
    n_middle = int(middle_row["n_datasets"]) if middle_row is not None else None

    note = f"Mean $\\pm$ SE across {n_first} datasets (First/Last)" if n_first is not None else ""
    if n_middle is not None and n_middle != n_first:
        note += f", {n_middle} for Middle (datasets with $L=2$ contribute no middle layer)"
    note += "."
    if exclusion_note:
        note += " " + exclusion_note
    return note


# ── table builders ───────────────────────────────────────────────────────────
def _table_minipage(df, filters, vmax, registry, width):
    lines = [r"\begin{minipage}{" + width + "}"]
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{l" + "c" * len(BIN_ORDER) + "}")
    lines.append(r"\toprule")
    lines.append("Checkpoint & " + " & ".join(BIN_LABEL[b] for b in BIN_ORDER) + r" \\")
    lines.append(r"\midrule")
    for ckpt in CHECKPOINT_ORDER:
        cells = [_cell_colored(_lookup(df, ckpt, b, **filters), vmax, registry) for b in BIN_ORDER]
        lines.append(f"{CHECKPOINT_LABEL[ckpt]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{minipage}")
    return lines


def _colorbar_minipage(vmax, registry, width):
    lines = [r"\begin{minipage}{" + width + "}"]
    lines.append(r"\centering")
    lines.append(_colorbar_tikz(vmax, registry))
    lines.append(r"\end{minipage}")
    return lines


def emit_single_reference_table(df, filters, caption, label, out_name, exclusion_note=None):
    """One reference (topological): table + one colorbar, side by side."""
    registry = _ColorRegistry(label)
    vals = _collect_means(df, filters)
    vmax = max(vals) if vals else 1.0

    table_mp    = _table_minipage(df, filters, vmax, registry, "0.78\\textwidth")
    colorbar_mp = _colorbar_minipage(vmax, registry, "0.15\\textwidth")

    lines = [r"\begin{table}[H]", r"\centering"]
    lines.append(registry.definitions())
    lines.extend(_hstack([table_mp, colorbar_mp]))

    pooled_se = _avg_pooled_se(df, **filters)
    coverage  = _coverage_note(df, filters, exclusion_note)
    shading   = (f" Cell shading: white = 0, dark red = {vmax:.2f} "
                 "(max value in this table), YlOrRd colormap, same scale as "
                 "Fig.~\\ref{fig: RM_Norm} etc.")
    lines.append(
        r"\caption{" + caption + " " + coverage +
        f" (Pooled fold/layer SE averages {pooled_se:.3f} across cells; not shown "
        "per cell, see text.)" + shading + "}"
    )
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def emit_two_reference_table(df, filters, caption, label, out_name, exclusion_note=None):
    """Two references (topological, zero): two independent [table + colorbar]
    units side by side, left to right, each with its own vmax."""
    registry = _ColorRegistry(label)

    def _block(reference, title, table_w, cbar_w):
        ref_filters = {**filters, "reference": reference}
        vals = _collect_means(df, ref_filters)
        vmax = max(vals) if vals else 1.0
        table_mp = _table_minipage(df, ref_filters, vmax, registry, table_w)
        table_mp = [r"\begin{minipage}{" + table_w + "}", r"\centering",
                    r"\footnotesize " + title + r"\\[3pt]"] + table_mp[2:]
        colorbar_mp = _colorbar_minipage(vmax, registry, cbar_w)
        return table_mp, colorbar_mp

    top_table_mp, top_cbar_mp   = _block("topological", r"vs.\ topological", "0.34\\textwidth", "0.085\\textwidth")
    zero_table_mp, zero_cbar_mp = _block("zero",        r"vs.\ zero",        "0.34\\textwidth", "0.085\\textwidth")

    lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize"]
    lines.append(registry.definitions())
    lines.extend(_hstack([top_table_mp, top_cbar_mp, zero_table_mp, zero_cbar_mp]))

    coverage = _coverage_note(df, {**filters, "reference": "topological"}, exclusion_note)
    shading  = (" Cell shading (each block independently scaled): white = 0, "
                "dark red = block maximum, YlOrRd colormap.")
    lines.append(r"\caption{" + caption + " " + coverage + shading + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main():
    vs_ref = pd.read_csv(AGG_DIR / "forman_wasserstein_agg_vs_reference.csv")
    cross  = pd.read_csv(AGG_DIR / "forman_wasserstein_agg_cross.csv")

    print("Table 1 — GeneralSheaf, normalised=True")
    emit_single_reference_table(
        vs_ref, dict(model="GeneralSheaf", normalised=True, use_other=False, reference="topological"),
        "Global Wasserstein alignment, GeneralSheaf, normalised Laplacian, vs.\\ topological reference.",
        "tab:global_general_norm", "table1_general_norm.tex")

    print("Table 2 — GeneralSheaf, normalised=False")
    emit_two_reference_table(
        vs_ref, dict(model="GeneralSheaf", normalised=False, use_other=False),
        "Global Wasserstein alignment, GeneralSheaf, unnormalised Laplacian.",
        "tab:global_general_unnorm", "table2_general_unnorm.tex")

    print("Table 3 — JointSheafParams (identity init), normalised=False")
    emit_two_reference_table(
        vs_ref, dict(model="JointSheafParams", map_type="identity", normalised=False, use_other=False),
        "Global Wasserstein alignment, JointSheafParams (identity init), unnormalised.",
        "tab:global_joint_unnorm", "table3_joint_unnorm.tex")

    print("Table 4 — JointSheafParams (identity init), normalised=True")
    emit_single_reference_table(
        vs_ref, dict(model="JointSheafParams", map_type="identity", normalised=True, use_other=False, reference="topological"),
        "Global Wasserstein alignment, JointSheafParams (identity init), normalised.",
        "tab:global_joint_norm", "table4_joint_norm.tex",
        exclusion_note="Tolokers is excluded: training was unstable for this "
                        "configuration (JointSheafParams, normalised=True) and no "
                        "reference profile was saved.")

    print("Table 5 — GeneralSheaf, other Laplacian (trained False, normalised content)")
    emit_single_reference_table(
        vs_ref, dict(model="GeneralSheaf", normalised=False, use_other=True, reference="topological"),
        "Global Wasserstein alignment, GeneralSheaf other Laplacian: maps trained \\texttt{norm=False}, evaluated normalised.",
        "tab:global_general_other_norm", "table5_general_other_norm.tex")

    print("Table 6 — GeneralSheaf, other Laplacian (trained True, unnormalised content)")
    emit_two_reference_table(
        vs_ref, dict(model="GeneralSheaf", normalised=True, use_other=True),
        "Global Wasserstein alignment, GeneralSheaf other Laplacian: maps trained \\texttt{norm=True}, evaluated unnormalised.",
        "tab:global_general_other_unnorm", "table6_general_other_unnorm.tex")

    print("Table 7 — Cross-comparison Direction A")
    emit_single_reference_table(
        cross, dict(model="GeneralSheaf", direction="A", reference="cross"),
        "Global cross-comparison, Direction A (both normalised, different training objective).",
        "tab:global_cross_a", "table7_cross_a.tex")

    print("Table 8 — Cross-comparison Direction B")
    emit_single_reference_table(
        cross, dict(model="GeneralSheaf", direction="B", reference="cross"),
        "Global cross-comparison, Direction B (both unnormalised, different training objective).",
        "tab:global_cross_b", "table8_cross_b.tex")


if __name__ == "__main__":
    main()

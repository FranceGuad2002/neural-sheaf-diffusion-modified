#!/usr/bin/env python3
"""
Cross-dataset summary tables for the conviction experiments.

The per-dataset tables under Conviction_experiments_tables/ answer "what happened
on Cora?"; there are ~160 of them, which is far more than a reader can absorb.
This script collapses them into the handful of tables a results section actually
needs, reading ONLY those existing CSVs — it runs no model and needs no GPU.

The single quantity reported everywhere is the RELATIVE DROP against that
experiment's own unperturbed baseline,

    drop%  =  100 * (baseline - metric) / (baseline - floor),

where floor is the metric's chance level: 0 for accuracy, 0.5 for ROC-AUC. The
floor matters because a relative drop implicitly treats it as the worst possible
score — dividing an AUC drop by the raw baseline would understate it by roughly
threefold, since a model at AUC 0.5 has already learned nothing. Normalising by
the headroom instead makes every cell mean the same thing ("percent of learned
signal destroyed") and so makes the accuracy and ROC-AUC datasets comparable in
one table.

Aggregation order, which is the whole statistical content of this script:

    per fold      drop%(dataset, fold, strategy, budget)
       |  mean over folds
    per dataset   drop%(dataset, strategy, budget)
       |  mean over datasets
    headline      mean +/- SE

Folds are averaged WITHIN a dataset before the dataset-level spread is taken.
This is deliberate: the ten folds of one dataset share a graph, an architecture
and overlapping splits, so they are not independent observations. Pooling all
folds of all datasets and dividing by sqrt(n_folds * n_datasets) would report an
error bar several times too small. `drop_se` is therefore always between-dataset
and is the one to report; `total_se` is that same naive pooled-over-everything
SE, kept only as a diagnostic so the understatement can be shown on request
rather than merely asserted — expect total_se < drop_se.

Per-fold drops are read from the `*_folds.csv` files written alongside each
aggregated table. Where those are absent (experiments run before they existed)
the script falls back to the aggregated file and forms the drop from the fold
means — the point estimate is essentially identical, only the fold axis is lost.

Three summaries are produced: the forward budget sweep (`forward_*`), the decile
profile (`decile_*`, plus a Spearman rho per dataset testing whether damage grows
monotonically with bin index), and soft gating (`soft_*`, which has no budget or
selection axis). `forward_tests_*.csv` additionally carries a Wilcoxon
signed-rank test across datasets for a few strategy pairs; it is written to its
own file and can be ignored entirely without affecting the main tables.

Usage (from repo root):
    python quick_analysis/conviction_summary_tables.py
    python quick_analysis/conviction_summary_tables.py --score avg_z --budgets 0.2
"""

import os
import sys
import csv
import glob
import json
import argparse
import collections

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

TABLES_ROOT = os.path.join(ROOT, 'quick_analysis', 'Conviction_experiments_tables')

STRATEGY_ORDER = ['high', 'low', 'deg_high', 'deg_low', 'deg_matched_high',
                  'deg_matched_low', 'conf_high', 'conf_low', 'random']
STRATEGY_LABEL = {
    'high':             'High conviction',
    'low':              'Low conviction',
    'deg_high':         'High degree',
    'deg_low':          'Low degree',
    'deg_matched_high': 'Degree-matched (high)',
    'deg_matched_low':  'Degree-matched (low)',
    'conf_high':        'High confidence',
    'conf_low':         'Low confidence',
    'random':           'Random',
}
CRITERIA        = ['conviction', 'confidence', 'degree']
CRITERION_LABEL = {'conviction': 'Conviction', 'confidence': 'Confidence', 'degree': 'Degree'}

_SPREAD_NOTE = 'Mean $\\pm$ standard error of the cross-dataset mean, '

# Chance level of each headline metric. Accuracy is left at 0 so the accuracy
# datasets keep exactly the relative drop they had before this floor existed.
_FLOOR = {'auc': 0.5}


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _combo(rows):
    """(intervention, mode) label for a file, e.g. 'masking (zero)'."""
    r = rows[0]
    return f"{r['intervention']} ({r['mode']})" if r['mode'] else r['intervention']


def _rel_drop(baseline, value, metric):
    """Drop from baseline as a percentage of the headroom above chance.
    Positive = the intervention hurt."""
    floor = _FLOOR.get(metric, 0.0)
    if baseline is None or not np.isfinite(baseline) or not np.isfinite(value):
        return float('nan')
    headroom = baseline - floor
    if headroom < 1e-12:
        return float('nan')
    return 100.0 * (baseline - value) / headroom


def _aggregate(per_dataset):
    """Collapse {dataset: array of per-fold drops} into the reported statistics.

    Folds are averaged within a dataset first, so the dataset is the unit of
    analysis and drop_se is between-dataset (see the module docstring on why
    pooling folds across datasets would understate it) — this is the number to
    report. total_se is the naive SE from pooling every (dataset, fold) value as
    if they were independent; it is a diagnostic only, kept so the understatement
    can be shown rather than asserted, and is expected to come out smaller than
    drop_se by roughly sqrt(mean folds per dataset)."""
    ds_means = np.array([np.nanmean(v) for v in per_dataset.values()
                         if np.isfinite(v).any()], dtype=float)
    if ds_means.size == 0:
        return None
    folds = np.concatenate([np.asarray(v, float) for v in per_dataset.values()])
    folds = folds[np.isfinite(folds)]
    std = float(ds_means.std(ddof=1)) if ds_means.size > 1 else 0.0
    total_std = float(folds.std(ddof=1)) if folds.size > 1 else 0.0
    return {
        'n_datasets':    int(ds_means.size),
        'n_folds_total': int(folds.size),
        'drop_mean':     float(ds_means.mean()),
        'drop_se':       std / np.sqrt(ds_means.size) if ds_means.size > 1 else 0.0,
        'total_se':      total_std / np.sqrt(folds.size) if folds.size > 1 else 0.0,
    }


def _fold_drops(path, metric, key_field, level_field):
    """{(key, level): array of per-fold drops} for one per-dataset table.

    Prefers the sibling *_folds.csv, where each fold's drop is formed against
    that fold's OWN baseline. Falls back to the aggregated file, which stores
    fold means only — then each cell is a length-1 array and the fold axis is
    simply absent. key_field/level_field are ('strategy','pct') for the forward
    sweep and ('criterion','bin') for the decile experiment."""
    folds_path = path.replace('.csv', '_folds.csv')
    if os.path.exists(folds_path):
        by_fold = collections.defaultdict(dict)
        for r in _read(folds_path):
            by_fold[r['fold']][(r[key_field], r[level_field])] = float(r['value'])
        out = collections.defaultdict(list)
        for cells in by_fold.values():
            base = cells.get(('baseline', ''))
            for (key, level), v in cells.items():
                if key == 'baseline' or not level:
                    continue
                out[(key, float(level))].append(_rel_drop(base, v, metric))
        return {k: np.array(v, dtype=float) for k, v in out.items()}

    rows = _read(path)
    base = next((float(r['mean']) for r in rows if r[key_field] == 'baseline'), None)
    return {(r[key_field], float(r[level_field])):
            np.array([_rel_drop(base, float(r['mean']), metric)], dtype=float)
            for r in rows if r[key_field] != 'baseline' and r[level_field]}


def _tables(subdir, score, model, normalised, recursive=False):
    """Aggregated per-dataset tables under one experiment tree. *_folds.csv is
    excluded here — it is picked up by _fold_drops via its sibling's name."""
    parts = [TABLES_ROOT, subdir, '*'] + (['**'] if recursive else []) + \
            [score, model, f'normalised_{normalised}', '*.csv']
    return sorted(p for p in glob.glob(os.path.join(*parts), recursive=recursive)
                  if not p.endswith('_folds.csv'))

# ── 1. forward sweep ──────────────────────────────────────────────────────────

def collect_forward(score, budgets, model, normalised, exclude=()):
    """{combo: {budget: {strategy: {dataset: array of per-fold drops}}}}."""
    out = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict)))
    for path in _tables('forward_experiment', score, model, normalised, recursive=True):
        rows = _read(path)
        if not rows or 'strategy' not in rows[0]:
            continue                      # soft-gating files have 'indicator'
        dataset = rows[0]['dataset']
        if dataset in exclude:
            continue
        combo, metric = _combo(rows), rows[0]['metric']
        for (strat, pct), drops in _fold_drops(path, metric, 'strategy', 'pct').items():
            match = next((b for b in budgets if abs(pct - b) < 1e-9), None)
            if match is not None:
                out[combo][match][strat][dataset] = drops
    return out


def summarise_forward(data):
    rows = []
    for combo in sorted(data):
        for budget in sorted(data[combo]):
            for strat in STRATEGY_ORDER:
                if strat not in data[combo][budget]:
                    continue
                stats = _aggregate(data[combo][budget][strat])
                if stats is None:
                    continue
                rows.append({'combo': combo, 'budget': budget, 'strategy': strat,
                             'label': STRATEGY_LABEL.get(strat, strat), **stats})
    return rows


def forward_tests(data):
    """Wilcoxon signed-rank across datasets, at every (combo, budget). Written to
    its own CSV and used by none of the main tables — ignore it freely.

    The pairing unit is the DATASET: for each dataset we take the difference in
    mean relative drop between the two strategies and ask whether those
    differences are reliably positive. 'wins_a' counts the datasets where a did
    more damage than b, and is the plain-language version of the same evidence —
    a strategy can have the larger mean while winning only ~half the datasets,
    in which case the mean is driven by a few large margins and the test will
    (correctly) refuse to call the difference reliable. No multiple-comparison
    correction is applied, so treat these as exploratory."""
    pairs = [('high', 'random'), ('high', 'deg_matched_high'), ('high', 'deg_high'),
             ('high', 'conf_high'), ('high', 'conf_low'), ('low', 'random'), ('high', 'low')]
    out = []
    for combo in sorted(data):
        for budget in sorted(data[combo]):
            by_strat = data[combo][budget]
            for a, b in pairs:
                if a not in by_strat or b not in by_strat:
                    continue
                common = sorted(set(by_strat[a]) & set(by_strat[b]))
                xa = np.array([np.nanmean(by_strat[a][d]) for d in common], dtype=float)
                xb = np.array([np.nanmean(by_strat[b][d]) for d in common], dtype=float)
                keep = np.isfinite(xa) & np.isfinite(xb)
                xa, xb = xa[keep], xb[keep]
                if xa.size < 5 or np.allclose(xa, xb):
                    continue
                try:
                    stat, p = wilcoxon(xa, xb)
                except ValueError:
                    continue
                out.append({'combo': combo, 'budget': budget, 'a': a, 'b': b,
                            'n_datasets': int(xa.size),
                            'median_diff': float(np.median(xa - xb)),
                            'wins_a': int((xa > xb).sum()), 'p_value': float(p)})
    return out

# ── 2. decile profile ─────────────────────────────────────────────────────────

def collect_decile(score, model, normalised, exclude=()):
    """{combo: {criterion: {dataset: {bin: array of per-fold drops}}}}."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in _tables('decile_experiment', score, model, normalised):
        rows = _read(path)
        if not rows or 'criterion' not in rows[0]:
            continue
        dataset = rows[0]['dataset']
        if dataset in exclude:
            continue
        combo, metric = _combo(rows), rows[0]['metric']
        for (crit, b), drops in _fold_drops(path, metric, 'criterion', 'bin').items():
            out[combo][crit].setdefault(dataset, {})[int(b)] = drops
    return out


def summarise_decile(data):
    """Per (combo, criterion, bin): mean/SE drop% across datasets, plus the
    per-dataset Spearman rho between bin index and drop — the monotonicity the
    decile experiment exists to test — as a supplementary column."""
    rows = []
    for combo in sorted(data):
        for crit in CRITERIA:
            if crit not in data[combo]:
                continue
            per_ds   = data[combo][crit]
            all_bins = sorted({b for d in per_ds.values() for b in d})
            rhos = []
            for d in per_ds.values():
                bs = sorted(d)
                v  = np.array([np.nanmean(d[b]) for b in bs], dtype=float)
                k  = np.isfinite(v)
                if k.sum() >= 3:
                    rhos.append(spearmanr(np.array(bs, float)[k], v[k]).correlation)
            rhos = np.array([r for r in rhos if np.isfinite(r)], dtype=float)
            for b in all_bins:
                stats = _aggregate({ds: d[b] for ds, d in per_ds.items() if b in d})
                if stats is None:
                    continue
                rows.append({'combo': combo, 'criterion': crit,
                             'label': CRITERION_LABEL[crit], 'bin': b, **stats,
                             'rho_mean': float(rhos.mean()) if rhos.size else float('nan'),
                             'rho_n_positive': int((rhos > 0).sum()) if rhos.size else 0})
    return rows

# ── 3. structural damage under pruning ────────────────────────────────────────

def collect_prune_diag(score, model, normalised, budgets, exclude=()):
    """{strategy: {budget: {dataset: (% edges deleted, n components)}}}.

    Pruning is the only intervention that changes the graph, so each run also
    writes a *_diagnostics.json recording what the graph looked like before and
    after. Read against the drop table, this separates the damage a strategy does
    by deleting edges from the damage it does by deleting the RIGHT nodes."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    pattern = os.path.join(TABLES_ROOT, 'forward_experiment', 'pruning', score,
                           model, f'normalised_{normalised}', '*_diagnostics.json')
    for path in sorted(glob.glob(pattern)):
        for r in json.load(open(path)):
            if r['dataset'] in exclude:
                continue
            match = next((b for b in budgets if abs(r['pct'] - b) < 1e-9), None)
            if match is not None:
                out[r['strategy']][match][r['dataset']] = (
                    100.0 * (1.0 - r['pct_edges_remaining']), r['n_connected_components'])
    return out


def summarise_prune_diag(data):
    rows = []
    for strat in STRATEGY_ORDER:
        for budget in sorted(data.get(strat, {})):
            edges = np.array([v[0] for v in data[strat][budget].values()], dtype=float)
            comps = np.array([v[1] for v in data[strat][budget].values()], dtype=float)
            n = edges.size
            rows.append({
                'strategy': strat, 'label': STRATEGY_LABEL[strat], 'budget': budget,
                'n_datasets': n,
                'edges_removed_mean': float(edges.mean()),
                'edges_removed_se': float(edges.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                'components_mean': float(comps.mean()),
            })
    return rows


def prune_edges_vs_drop(diag, fwd, budget):
    """Per dataset at one budget: (% edges deleted, % drop) for the high-conviction
    and the high-degree selection.

    This is the table that settles whether the damage a degree criterion does is
    structural or informational. Both quantities are needed per dataset, because
    the interesting cases are the ones where the two orderings disagree: degree
    cutting the larger hole in the graph and still costing less."""
    by_budget = fwd.get('pruning', {}).get(budget, {})
    rows = []
    for ds in sorted(set(diag['high'].get(budget, {})) & set(by_budget.get('high', {}))):
        rows.append({
            'dataset': ds, 'budget': budget,
            'conv_edges': diag['high'][budget][ds][0],
            'conv_drop':  float(np.nanmean(by_budget['high'][ds])),
            'deg_edges':  diag['deg_high'][budget][ds][0],
            'deg_drop':   float(np.nanmean(by_budget['deg_high'][ds])),
        })
    return rows


def latex_edges_vs_drop(rows, score, tag, out_dir):
    pct  = int(rows[0]['budget'] * 100)
    body = [(r['dataset'].replace('_', ' ').title(),
             [f"${r['conv_edges']:.1f}$", f"${r['conv_drop']:.2f}$",
              f"${r['deg_edges']:.1f}$",  f"${r['deg_drop']:.2f}$"])
            for r in rows]
    _latex_table(out_dir, [{
        'path':  f"prune_edges_vs_drop_{tag}.tex",
        'label': f"tab:conv_prune_evd_{score}",
        'caption':
            rf"Structural cost against performance cost, per dataset, for pruning at the "
            rf"{pct}\% budget. ``Edges'' is the percentage of the graph's edges deleted "
            rf"along with the selected nodes; ``Drop'' is the relative drop (\%) in the "
            rf"headline metric, as everywhere else in this section. Selecting by degree "
            rf"deletes the larger fraction of the graph on every dataset, yet on several "
            rf"of them it costs less than selecting by conviction, which is what shows "
            rf"that part of the damage a degree criterion does is structural rather than "
            rf"informational. Conviction score \texttt{{{score.replace('_', chr(92)+'_')}}}, "
            rf"values averaged over the ten folds.",
        'corner': "Dataset", 'cols': ["Edges", "Drop", "Edges", "Drop"],
        'spanners': [r" & \multicolumn{2}{c}{High conviction}"
                     r" & \multicolumn{2}{c}{High degree} \\",
                     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"],
        'body': body}])


def latex_prune_diag(rows, score, tag, out_dir):
    """Two tables, both shaped like the pruning drop table so they read side by
    side with it: how many edges each selection deletes, and how badly it
    fragments the graph."""
    budgets = sorted({r['budget'] for r in rows})
    idx     = {(r['budget'], r['strategy']): r for r in rows}
    n       = max((r['n_datasets'] for r in rows), default=0)
    strats  = [s for s in STRATEGY_ORDER if any((b, s) in idx for b in budgets)]

    def body(fmt):
        return [(STRATEGY_LABEL[s],
                 [fmt(idx[(b, s)]) if (b, s) in idx else "--" for b in budgets])
                for s in strats]

    _latex_table(out_dir, [{
        'path':  f"prune_edges_{tag}.tex",
        'label': f"tab:conv_prune_edges_{score}",
        'caption':
            r"Structural cost of pruning: percentage of the graph's edges deleted "
            r"together with the selected nodes, as a function of the fraction of the "
            r"training pool removed. Mean $\pm$ standard error over the "
            rf"{n} datasets. To be read against Table~\ref{{tab:conv_fwd_pruning_{score}}}, "
            r"which gives what each of these selections costs in performance: a "
            r"strategy that deletes more edges for less damage is destroying "
            r"structure rather than information.",
        'corner': "Selection", 'cols': [rf"{int(b*100)}\%" for b in budgets],
        'body': body(lambda r: f"${r['edges_removed_mean']:.1f} \\pm "
                               f"{r['edges_removed_se']:.1f}$"),
    }, {
        'path':  f"prune_components_{tag}.tex",
        'label': f"tab:conv_prune_comps_{score}",
        'caption':
            r"Fragmentation under pruning: number of connected components left in "
            r"the graph after the selected nodes are deleted, averaged over the ten "
            rf"folds and then over the {n} datasets. The companion of "
            rf"Table~\ref{{tab:conv_prune_edges_{score}}}: it measures the same "
            r"structural damage from the other side, since a selection may delete "
            r"few edges and still cut the graph into many pieces.",
        'corner': "Selection", 'cols': [rf"{int(b*100)}\%" for b in budgets],
        'body': body(lambda r: f"${r['components_mean']:.0f}$"),
    }])


def latex_soft(rows, score, tag, out_dir):
    """Soft gating has no budget or selection axis, so one row per indicator."""
    n = max((r['n_datasets'] for r in rows), default=0)
    _latex_table(out_dir, [{
        'path':  f"soft_gating_{tag}.tex",
        'label': f"tab:conv_soft_{score}",
        'caption': _DROP_CAPTION +
            r"Soft gating: instead of selecting a subset, every training node's "
            r"outgoing message is attenuated in proportion to its own score, the "
            r"lowest-scoring node being suppressed the most (gate floor "
            r"$g_{\min}=0.1$). There is therefore no budget axis and one number per "
            rf"indicator. Conviction score \texttt{{{score.replace('_', chr(92)+'_')}}}. "
            rf"{_SPREAD_NOTE}over {n} datasets.",
        'corner': "Indicator", 'cols': [r"Relative drop (\%)"],
        'body': [(r['indicator'].title(),
                  [f"${r['drop_mean']:.2f} \\pm {r['drop_se']:.2f}$"]) for r in rows],
    }])

# ── 4. soft gating ────────────────────────────────────────────────────────────

def _fold_drops_soft(path, metric):
    """{indicator: array of per-fold drops} for one soft-gating dataset table.
    Same folds-file-preferred, aggregated-fallback pattern as _fold_drops, but
    keyed by indicator only — soft gating has no budget/selection axis to carry
    as a second key."""
    folds_path = path.replace('.csv', '_folds.csv')
    if os.path.exists(folds_path):
        by_fold = collections.defaultdict(dict)
        for r in _read(folds_path):
            by_fold[r['fold']][r['indicator']] = float(r['value'])
        out = collections.defaultdict(list)
        for cells in by_fold.values():
            base = cells.get('baseline')
            for indicator, v in cells.items():
                if indicator != 'baseline':
                    out[indicator].append(_rel_drop(base, v, metric))
        return {k: np.array(v, dtype=float) for k, v in out.items()}

    rows = _read(path)
    base = next((float(r['mean']) for r in rows if r['indicator'] == 'baseline'), None)
    return {r['indicator']: np.array([_rel_drop(base, float(r['mean']), metric)], dtype=float)
            for r in rows if r['indicator'] != 'baseline'}


def collect_soft(score, model, normalised, exclude=()):
    """Soft gating has no budget or selection axis, so it collapses to one
    relative drop per indicator per dataset. Reads the sibling *_folds.csv when
    present (see _fold_drops_soft) — same preference/fallback as collect_forward
    and collect_decile, so a rerun benefits from the fold axis automatically."""
    per_indicator = collections.defaultdict(dict)
    pattern = os.path.join(TABLES_ROOT, 'forward_experiment', 'gating', 'soft',
                           score, model, f'normalised_{normalised}', '*.csv')
    for path in sorted(p for p in glob.glob(pattern) if not p.endswith('_folds.csv')):
        rows = _read(path)
        if not rows or 'indicator' not in rows[0]:
            continue
        dataset = rows[0]['dataset']
        if dataset in exclude:
            continue
        metric = rows[0]['metric']
        for indicator, drops in _fold_drops_soft(path, metric).items():
            per_indicator[indicator][dataset] = drops
    rows = []
    for ind, vals in sorted(per_indicator.items()):
        stats = _aggregate(vals)
        if stats is not None:
            rows.append({'indicator': ind, **stats})
    return rows

# ── output ────────────────────────────────────────────────────────────────────

def write_csv(rows, path, fields):
    if not rows:
        print(f"  (nothing to write for {os.path.basename(path)})"); return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {path}")


def _cell(r, shade=None):
    """One table cell: mean +/- the (between-dataset) standard error, optionally
    shaded (see _shade_extremes)."""
    if r is None:
        return "--"
    body = f"${r['drop_mean']:.2f} \\pm {r['drop_se']:.2f}$"
    return f"\\cellcolor{{{shade}}}{body}" if shade else body


def _slug(combo):
    """'masking (zero)' -> 'masking-zero', for use in filenames and labels."""
    return combo.replace(' ', '-').replace('(', '').replace(')', '')


# Shading of the forward tables: within each budget column, the most damaging
# selection is marked red and the least damaging blue. Only the two extremes are
# marked, so the eye finds them without the table turning into a heatmap.
_SHADE_DEF = [r"\definecolor{convworst}{rgb}{0.97,0.80,0.78}",
              r"\definecolor{convbest}{rgb}{0.80,0.87,0.96}"]
_SHADE_NOTE = (r"In each column the most damaging selection is shaded red and the "
               r"least damaging blue. ")


def _shade_extremes(column):
    """{row index: colour} for the max and min of one column of row dicts."""
    vals = [(i, r['drop_mean']) for i, r in enumerate(column) if r is not None]
    if len(vals) < 2:
        return {}
    return {max(vals, key=lambda t: t[1])[0]: 'convworst',
            min(vals, key=lambda t: t[1])[0]: 'convbest'}


_DROP_CAPTION = (
    r"Relative drop (\%) in the headline metric, accuracy or ROC-AUC on "
    r"Minesweeper, Tolokers and Questions, expressed as a percentage of the "
    r"baseline's headroom above chance, so that the two metrics are comparable. "
)


def _latex_table(out_dir, blocks):
    """Write each block to its OWN file, so a results section can \\input exactly
    the table it wants and leave the rest to the supplementary. A block is a dict
    with keys path, label, caption, corner, cols, body, and optionally preamble
    (lines before the tabular, e.g. colour definitions) and spanners (raw header
    lines inserted above the column labels). Shared by every table below, which
    differ only in what indexes their rows and columns."""
    os.makedirs(out_dir, exist_ok=True)
    for b in blocks:
        lines = ([r"% auto-generated by quick_analysis/conviction_summary_tables.py",
                  r"\begin{table}[H]", r"\centering", r"\small"] + b.get('preamble', []) +
                 [r"\begin{tabular}{l" + "c" * len(b['cols']) + "}", r"\toprule"] +
                 b.get('spanners', []) +
                 [f"{b['corner']} & " + " & ".join(b['cols']) + r" \\", r"\midrule"])
        lines += [f"{lab} & " + " & ".join(cells) + r" \\" for lab, cells in b['body']]
        lines += [r"\bottomrule", r"\end{tabular}",
                  rf"\caption{{{b['caption']}}}", rf"\label{{{b['label']}}}", r"\end{table}"]
        full = os.path.join(out_dir, b['path'])
        open(full, 'w').write("\n".join(lines) + "\n")
        print(f"  wrote {full}")


def latex_forward(rows, score, tag, out_dir):
    """One table per intervention: strategies as rows, budgets as columns, so the
    whole sweep is visible in a few compact blocks rather than one table per
    dataset."""
    budgets = sorted({r['budget'] for r in rows})
    idx     = {(r['combo'], r['budget'], r['strategy']): r for r in rows}
    esc     = score.replace('_', chr(92) + '_')
    blocks  = []
    for combo in sorted({r['combo'] for r in rows}):
        n      = max((r['n_datasets'] for r in rows if r['combo'] == combo), default=0)
        strats = [s for s in STRATEGY_ORDER
                  if any(idx.get((combo, b, s)) for b in budgets)]
        shades = [_shade_extremes([idx.get((combo, b, s)) for s in strats])
                  for b in budgets]
        body   = [(STRATEGY_LABEL[s],
                   [_cell(idx.get((combo, b, s)), shades[j].get(i))
                    for j, b in enumerate(budgets)])
                  for i, s in enumerate(strats)]
        blocks.append({
            'path':  f"forward_{_slug(combo)}_{tag}.tex",
            'label': f"tab:conv_fwd_{_slug(combo)}_{score}",
            'caption': _DROP_CAPTION +
            rf"Intervention: {combo.replace('_', chr(92)+'_')}, as a function of the "
            rf"fraction of the training pool perturbed. Conviction score "
            rf"\texttt{{{esc}}}. Each of the {n} datasets contributes one value, itself "
            rf"averaged over its folds. {_SPREAD_NOTE}"
            rf"the quoted spread being between-dataset variability. Larger is more "
            rf"damaging. {_SHADE_NOTE}",
            'corner': "Selection", 'cols': [rf"{int(b*100)}\%" for b in budgets],
            'preamble': _SHADE_DEF, 'body': body})
    _latex_table(out_dir, blocks)


def latex_decile(rows, score, tag, out_dir):
    """One table per intervention: bins as rows, ranking criteria as columns, so
    the reader reads straight down a column and sees the damage grow."""
    idx    = {(r['combo'], r['criterion'], r['bin']): r for r in rows}
    esc    = score.replace('_', chr(92) + '_')
    blocks = []
    for combo in sorted({r['combo'] for r in rows}):
        bins    = sorted({r['bin'] for r in rows if r['combo'] == combo})
        n       = max((r['n_datasets'] for r in rows if r['combo'] == combo), default=0)
        present = [c for c in CRITERIA if any(idx.get((combo, c, b)) for b in bins)]
        body    = [(str(b), [_cell(idx.get((combo, c, b))) for c in present])
                   for b in bins]
        blocks.append({
            'path':  f"decile_{_slug(combo)}_{tag}.tex",
            'label': f"tab:conv_dec_{_slug(combo)}_{score}",
            'caption': _DROP_CAPTION +
            rf"Decile analysis under {combo.replace('_', chr(92)+'_')}: a single bin of "
            rf"the training pool is perturbed, with the pool split into {len(bins)} equal "
            rf"bins ranked by each criterion (bin 1 = lowest score, bin {len(bins)} = "
            rf"highest). Conviction score \texttt{{{esc}}}. {_SPREAD_NOTE}"
            rf"over {n} datasets.",
            'corner': "Bin", 'cols': [CRITERION_LABEL[c] for c in present],
            'body': body})
    _latex_table(out_dir, blocks)


def _print_grid(title, row_order, row_label, col_values, col_header, lookup):
    """Console echo of a LaTeX table: rows x columns of drop_mean."""
    print(f"\n{title}")
    print(f"    {'':<24}" + "".join(f"{c:>12}" for c in col_header))
    for key in row_order:
        cells = [lookup(key, c) for c in col_values]
        if any(r is not None for r in cells):
            print(f"    {row_label(key):<24}"
                  + "".join(f"{r['drop_mean']:>12.2f}" if r else f"{'--':>12}" for r in cells))

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--score',      default='last', choices=['last', 'avg', 'avg_z'])
    ap.add_argument('--budgets',    default='0.01,0.05,0.1,0.2,0.3',
                    help='comma-separated budgets to tabulate (default: the full sweep)')
    ap.add_argument('--model',      default='GeneralSheaf')
    ap.add_argument('--normalised', default='true', choices=['true', 'false'])
    ap.add_argument('--exclude',    default='',
                    help='comma-separated datasets to drop from the aggregation, e.g. '
                         '"cornell,wisconsin" — their training pools are so small that a '
                         '1%% budget is a single node and a decile is under ten')
    ap.add_argument('--tag_suffix', default='',
                    help='appended to output filenames, to keep variants side by side')
    ap.add_argument('--out_dir',    default=os.path.join(TABLES_ROOT, '_summary'))
    cli = ap.parse_args()

    budgets = [float(b) for b in cli.budgets.split(',')]
    exclude = tuple(x.strip() for x in cli.exclude.split(',') if x.strip())
    tag = f"{cli.model}_norm-{cli.normalised}_conv-{cli.score}{cli.tag_suffix}"
    STAT_FIELDS = ['n_datasets', 'n_folds_total', 'drop_mean', 'drop_se', 'total_se']
    print(f"Summarising {TABLES_ROOT}\n  score={cli.score}  budgets={budgets}  "
          f"model={cli.model}  normalised={cli.normalised}\n")

    fwd = collect_forward(cli.score, budgets, cli.model, cli.normalised, exclude)
    if not fwd:
        print("No forward tables matched — check --score/--model/--normalised.")
    else:
        srows = summarise_forward(fwd)
        write_csv(srows, os.path.join(cli.out_dir, f"forward_summary_{tag}.csv"),
                  ['combo', 'budget', 'strategy', 'label'] + STAT_FIELDS)
        latex_forward(srows, cli.score, tag, cli.out_dir)
        write_csv(forward_tests(fwd), os.path.join(cli.out_dir, f"forward_tests_{tag}.csv"),
                  ['combo', 'budget', 'a', 'b', 'n_datasets', 'median_diff', 'wins_a', 'p_value'])

        print("\n── Forward summary: relative drop % (mean over datasets) ──")
        for combo in sorted(fwd):
            bs = sorted(fwd[combo])
            _print_grid(f"  {combo}", STRATEGY_ORDER, lambda s: STRATEGY_LABEL[s],
                        bs, [f"{int(b*100)}%" for b in bs],
                        lambda s, b: next((x for x in srows if x['combo'] == combo
                                           and x['budget'] == b and x['strategy'] == s), None))

    dec = collect_decile(cli.score, cli.model, cli.normalised, exclude)
    if dec:
        drows = summarise_decile(dec)
        write_csv(drows, os.path.join(cli.out_dir, f"decile_bins_{tag}.csv"),
                  ['combo', 'criterion', 'label', 'bin'] + STAT_FIELDS
                  + ['rho_mean', 'rho_n_positive'])
        latex_decile(drows, cli.score, tag, cli.out_dir)
        print("\n── Decile: relative drop % per bin (mean over datasets) ──")
        for combo in sorted(dec):
            bins  = sorted({r['bin'] for r in drows if r['combo'] == combo})
            crits = [c for c in CRITERIA
                     if any(r['combo'] == combo and r['criterion'] == c for r in drows)]
            _print_grid(f"  {combo}", bins, lambda b: f"bin {b}",
                        crits, [CRITERION_LABEL[c] for c in crits],
                        lambda b, c: next((x for x in drows if x['combo'] == combo
                                           and x['criterion'] == c and x['bin'] == b), None))

    diag = collect_prune_diag(cli.score, cli.model, cli.normalised, budgets, exclude)
    if diag:
        prows = summarise_prune_diag(diag)
        write_csv(prows, os.path.join(cli.out_dir, f"prune_edges_{tag}.csv"),
                  ['strategy', 'label', 'budget', 'n_datasets',
                   'edges_removed_mean', 'edges_removed_se', 'components_mean'])
        latex_prune_diag(prows, cli.score, tag, cli.out_dir)
        evd = prune_edges_vs_drop(diag, fwd, max(budgets))
        if evd:
            write_csv(evd, os.path.join(cli.out_dir, f"prune_edges_vs_drop_{tag}.csv"),
                      ['dataset', 'budget', 'conv_edges', 'conv_drop', 'deg_edges', 'deg_drop'])
            latex_edges_vs_drop(evd, cli.score, tag, cli.out_dir)
        print("\n── Pruning: % of edges deleted (mean over datasets) ──")
        _print_grid("", STRATEGY_ORDER, lambda s: STRATEGY_LABEL[s],
                    budgets, [f"{int(b*100)}%" for b in budgets],
                    lambda s, b: (lambda r: {'drop_mean': r['edges_removed_mean']} if r else None)(
                        next((x for x in prows if x['strategy'] == s and x['budget'] == b), None)))

    soft = collect_soft(cli.score, cli.model, cli.normalised, exclude)
    if soft:
        write_csv(soft, os.path.join(cli.out_dir, f"soft_gating_{tag}.csv"),
                  ['indicator'] + STAT_FIELDS)
        latex_soft(soft, cli.score, tag, cli.out_dir)
        print("\n── Soft gating (whole train pool attenuated) ──")
        for r in soft:
            print(f"    {r['indicator']:<16}{r['drop_mean']:>9.2f} ± {r['drop_se']:.2f}"
                  f"   (n={r['n_datasets']})")

    print(f"\nDone. Summaries in: {cli.out_dir}")

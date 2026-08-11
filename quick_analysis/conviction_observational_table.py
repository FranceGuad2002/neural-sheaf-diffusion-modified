#!/usr/bin/env python3
"""
Cross-dataset table of the conviction's observational profile.

The heatmaps under quick_analysis/Norm_Analysis/conviction_heatmap/ answer "what
does the conviction of Cora correlate with?"; there is one per dataset per model
and they average the folds away before anything is written to disk. This script
collapses the per-fold CSVs that conviction_heatmap.py now writes alongside them
into the single table the results section needs. It reads only those CSVs, runs
no model and needs no GPU.

The aggregation is deliberately the SAME one used for the intervention tables in
conviction_summary_tables.py, so the error bars in every table of the section
mean the same thing:

    per fold      rho(dataset, fold)          at one checkpoint and one layer
       |  mean over folds
    per dataset   rho(dataset)
       |  mean over datasets
    headline      mean +/- SE across datasets

Folds are averaged WITHIN a dataset first: the ten folds of one dataset share a
graph and an architecture, so they are correlated re-measurements, not
independent samples, and the dataset is the unit of analysis. `_aggregate` is
imported from conviction_summary_tables rather than reimplemented, so the two
families of tables cannot drift apart.

Spearman rho is averaged directly, without a Fisher z transform, to stay
consistent with the heatmaps, which already average rho over folds. This biases
slightly towards zero where |rho| is large, so do not read the third decimal.

Each configuration is reported on its own terms. The table is NOT an ablation:
the two configurations differ both in the restriction-map family and in the
normalisation, so no single factor can be credited for a difference between the
columns. What the table supports is a description of what each pipeline does.

Usage (from repo root), after conviction_heatmap.py has been run for both:
    python quick_analysis/conviction_observational_table.py
    python quick_analysis/conviction_observational_table.py --layer last --checkpoints epoch-0,best-epoch
"""

import os
import sys
import csv
import glob
import argparse
import collections

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conviction_summary_tables import _aggregate, _latex_table, _SPREAD_NOTE

FOLDS_ROOT = os.path.join(ROOT, 'quick_analysis', 'Conviction_experiments_tables',
                          'observational')

OBSERVABLE_ORDER = ['degree', 'betweenness', 'confidence', 'saliency',
                    'cohens_d_train', 'cohens_d_test']
OBSERVABLE_LABEL = {
    'degree':         r"Spearman $\rho$ with degree",
    'betweenness':    r"Spearman $\rho$ with betweenness",
    'confidence':     r"Spearman $\rho$ with confidence",
    'saliency':       r"Spearman $\rho$ with saliency",
    'cohens_d_train': r"$|$Cohen's $d|$, train split",
    'cohens_d_test':  r"$|$Cohen's $d|$, test split",
}
CHECKPOINT_LABEL = {'epoch-0': 'Untrained', 'best-epoch': 'Best epoch'}

MODEL_LABEL = {
    ('GeneralSheaf', 'true'):  r"General sheaf, $\widetilde{\Delta}_{\mathcal{F}}$",
    ('GeneralSheaf', 'false'): r"General sheaf, $\Delta_{\mathcal{F}}$",
    ('BundleSheaf',  'true'):  r"Bundle sheaf, $\widetilde{\Delta}_{\mathcal{F}}$",
    ('BundleSheaf',  'false'): r"Bundle sheaf, $\Delta_{\mathcal{F}}$",
}


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def collect(model, normalised, exclude=()):
    """{(observable, checkpoint, layer_key): {dataset: array of per-fold values}}.

    `layer_key` is both the absolute layer index and the string 'last', so that a
    caller can ask for the final diffusion layer without knowing each dataset's
    depth: depth differs per dataset, and X_L is the layer the `last` conviction
    score is read from, so it is the one the intervention experiments rank by."""
    pattern = os.path.join(FOLDS_ROOT, model, f'normalised_{normalised}',
                           '*_observational_folds.csv')
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(glob.glob(pattern)):
        rows = _read(path)
        if not rows:
            continue
        dataset = rows[0]['dataset']
        if dataset in exclude:
            continue
        last = max(int(r['layer']) for r in rows)
        for r in rows:
            value = float(r['value'])
            if not np.isfinite(value):
                continue
            layer = int(r['layer'])
            keys = [(r['observable'], r['checkpoint'], layer)]
            if layer == last:
                keys.append((r['observable'], r['checkpoint'], 'last'))
            for k in keys:
                out[k][dataset].append(value)
    return {k: {d: np.array(v, dtype=float) for d, v in ds.items()}
            for k, ds in out.items()}


def summarise(data, checkpoints, layer):
    rows = []
    for observable in OBSERVABLE_ORDER:
        for ckpt in checkpoints:
            per_dataset = data.get((observable, ckpt, layer))
            if not per_dataset:
                continue
            stats = _aggregate(per_dataset)
            if stats is not None:
                rows.append({'observable': observable, 'checkpoint': ckpt,
                             'layer': str(layer), **stats})
    return rows


def _cell(r):
    # _aggregate names its fields after the intervention tables it was written
    # for; here the same two numbers are a correlation and its spread.
    return "--" if r is None else f"${r['drop_mean']:.2f} \\pm {r['drop_se']:.2f}$"


def latex(per_config, checkpoints, layer, out_dir, path, label):
    """Rows are observables; each configuration contributes one column group,
    with one column per checkpoint."""
    configs = [c for c in per_config if per_config[c]]
    idx = {(c, r['observable'], r['checkpoint']): r
           for c in configs for r in per_config[c]}
    n = max((r['n_datasets'] for c in configs for r in per_config[c]), default=0)

    spanner, rule, col = "", "", 2
    for model, normalised in configs:
        spanner += (rf" & \multicolumn{{{len(checkpoints)}}}{{c}}"
                    rf"{{{MODEL_LABEL.get((model, normalised), model)}}}")
        rule    += rf"\cmidrule(lr){{{col}-{col + len(checkpoints) - 1}}}"
        col     += len(checkpoints)

    body = [(OBSERVABLE_LABEL[o],
             [_cell(idx.get((c, o, ck))) for c in configs for ck in checkpoints])
            for o in OBSERVABLE_ORDER
            if any((c, o, ck) in idx for c in configs for ck in checkpoints)]

    where = ("the last diffusion layer $X_L$" if layer == 'last'
             else f"diffusion layer $X_{{{layer}}}$")
    _latex_table(out_dir, [{
        'path': path, 'label': label,
        'caption':
            rf"Observational profile of the conviction at {where}. Each entry is a "
            rf"Spearman rank correlation between the conviction and the named node "
            rf"property, or the absolute Cohen's $d$ separating the conviction of "
            rf"correctly and incorrectly classified nodes, computed per fold, "
            rf"averaged over the folds of a dataset, and then averaged over the "
            rf"{n} datasets. {_SPREAD_NOTE}the quoted spread being between-dataset "
            rf"variability. ``Untrained'' is the epoch-$0$ checkpoint, where the sheaf "
            rf"is randomly initialised and nothing has been learnt, and it therefore "
            rf"acts as a control: an entry that is already large there is produced by "
            rf"the architecture rather than by training. The two configurations differ "
            rf"in the restriction-map family and in the normalisation at once, so the "
            rf"table describes what each pipeline does and is not an ablation of "
            rf"either factor.",
        'corner': "Observable",
        'cols': [CHECKPOINT_LABEL.get(ck, ck) for _ in configs for ck in checkpoints],
        'spanners': [spanner + r" \\", rule],
        'body': body,
    }])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default='GeneralSheaf:true,BundleSheaf:false',
                    help='comma-separated model:normalised pairs, one column group each')
    ap.add_argument('--checkpoints', default='epoch-0,best-epoch',
                    help='comma-separated checkpoints, one column per group')
    ap.add_argument('--layer', default='last',
                    help="'last' for the final diffusion layer X_L, or an integer index")
    ap.add_argument('--exclude', default='cornell,wisconsin',
                    help='datasets dropped from the aggregation')
    ap.add_argument('--tag', default='observational_summary')
    ap.add_argument('--label', default='tab:conv_observational')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'quick_analysis',
                                                      'conviction_tables'))
    cli = ap.parse_args()

    configs = [tuple(c.split(':')) for c in cli.configs.split(',') if c]
    ckpts   = [c for c in cli.checkpoints.split(',') if c]
    exclude = tuple(x.strip() for x in cli.exclude.split(',') if x.strip())
    layer   = 'last' if cli.layer == 'last' else int(cli.layer)

    print(f"Reading {FOLDS_ROOT}\n  configs={configs}  checkpoints={ckpts}  "
          f"layer={layer}  excluding={exclude}\n")

    per_config, all_rows = {}, []
    for model, normalised in configs:
        data = collect(model, normalised, exclude)
        if not data:
            print(f"  [WARNING] no per-fold CSVs for {model}/normalised_{normalised} — "
                  f"run conviction_heatmap.py for it first")
            per_config[(model, normalised)] = []
            continue
        rows = summarise(data, ckpts, layer)
        per_config[(model, normalised)] = rows
        for r in rows:
            all_rows.append({'model': model, 'normalised': normalised, **r})
        print(f"  {model}/normalised_{normalised}:")
        for r in rows:
            print(f"    {r['observable']:<16}{r['checkpoint']:<12}"
                  f"{r['drop_mean']:+7.3f} ± {r['drop_se']:.3f}  (n={r['n_datasets']})")

    if not all_rows:
        print("\nNothing to write — no per-fold CSVs found."); sys.exit(1)

    os.makedirs(cli.out_dir, exist_ok=True)
    csv_path = os.path.join(cli.out_dir, f"{cli.tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'normalised', 'observable',
                                          'checkpoint', 'layer', 'n_datasets',
                                          'n_folds_total', 'drop_mean', 'drop_se',
                                          'total_se'])
        w.writeheader(); w.writerows(all_rows)
    print(f"\n  wrote {csv_path}")

    latex(per_config, ckpts, layer, cli.out_dir, f"{cli.tag}.tex", cli.label)


    print(f"\nDone. Table in: {cli.out_dir}")

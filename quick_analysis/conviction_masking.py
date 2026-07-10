#!/usr/bin/env python3
"""
Conviction masking experiment (Alessio's protocol) — auto-discovery runner.

Train the SNN normally, then — without retraining — progressively zero the
features of different node subsets and measure accuracy / confidence drop on
the *remaining unmasked test nodes*.

Five masking strategies at each k% budget:
  high        — top-k% by X_L conviction          (deterministic)
  low         — bottom-k% by X_L conviction        (deterministic)
  deg_high    — top-k% by degree                   (deterministic)
  deg_matched — k random nodes matching high's degree dist (1 draw/fold)
  random      — k uniformly random nodes           (1 draw/fold)

Output folder:
    quick_analysis/accuracy_ranking/<model>/normalised_{true|false}/

Usage (from repo root):
    python quick_analysis/conviction_masking.py --model GeneralSheaf --normalised true
    python quick_analysis/conviction_masking.py --model BundleSheaf  --normalised false
"""

import sys
import os
import re
import glob
import json
import argparse
import collections

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.utils import degree as pyg_degree
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from exp.parser import get_parser

# ── constants ─────────────────────────────────────────────────────────────────

AUC_DATASETS = {'minesweeper', 'tolokers', 'questions'}

MASK_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30]

STRATEGIES  = ['high',      'low',       'deg_high',  'deg_matched',        'random']
LABELS      = ['High conv', 'Low conv',  'High deg',  'Deg-matched random', 'Random']
COLORS      = ['firebrick', 'steelblue', 'darkorange','goldenrod',          'gray']
LINESTYLES  = ['-',         '--',        '-',         ':',                  ':']
MARKERS     = ['o',         'o',         's',         'o',                  '^']


def _model_classes():
    from models.disc_models import (
        DiscreteDiagSheafDiffusion,
        DiscreteBundleSheafDiffusion,
        DiscreteGeneralSheafDiffusion,
        DiscreteVanillaDiffusion,
        DiscreteVanillaDiffusionAlt,
        DiscreteJointSheafDiffusionParams,
        DiscreteJointSheafDiffusionParamsAlt,
        DiscreteJointSheafVanillaDiffusion,
    )
    return {
        'DiagSheaf':           DiscreteDiagSheafDiffusion,
        'BundleSheaf':         DiscreteBundleSheafDiffusion,
        'GeneralSheaf':        DiscreteGeneralSheafDiffusion,
        'VanillaSheaf':        DiscreteVanillaDiffusion,
        'ConvSheaf':           DiscreteVanillaDiffusionAlt,
        'JointSheafParams':    DiscreteJointSheafDiffusionParams,
        'JointSheafParamsAlt': DiscreteJointSheafDiffusionParamsAlt,
        'JointSheafVanilla':   DiscreteJointSheafVanillaDiffusion,
    }

_JOINT_MODELS = {'JointSheafParams', 'JointSheafParamsAlt'}
_TYPE_SHORT   = {'diagonal': 'diag', 'bundle': 'bundle', 'general': 'general'}
EXCLUDE       = {'synthetic_exp'}

# ── path helpers ──────────────────────────────────────────────────────────────

def _map_tag(args):
    if args.get('model') not in _JOINT_MODELS:
        return ""
    if not args.get('learn_first_maps', False):
        return "_first_maps_identity"
    short = _TYPE_SHORT.get(args.get('learnt_map_type', 'general'), 'general')
    return f"_first_maps_{short}"


def _base_dir(result_type, args):
    return os.path.join(
        ROOT, 'results', result_type,
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    )


def _pt_filename(args, fold):
    return f"{args['model']}_{args['dataset']}{_map_tag(args)}_fold{fold}_seed{args['seed']}.pt"


def _weights_path(args, fold):
    fn = _pt_filename(args, fold).replace('.pt', '.pth')
    return os.path.join(_base_dir('model_weights', args), 'checkpoints', 'best-epoch', fn)


def _masks_path(args, fold):
    fn = _pt_filename(args, fold).replace('.pt', '_masks.pt')
    return os.path.join(_base_dir('model_weights', args), 'checkpoints', 'best-epoch', fn)


def _preds_path(args, fold):
    return os.path.join(_base_dir('node_preds', args), _pt_filename(args, fold))


def _norms_path(args, fold):
    return os.path.join(_base_dir('node_norms', args), 'checkpoints', 'best-epoch',
                        _pt_filename(args, fold))

# ── loading helpers ───────────────────────────────────────────────────────────

def _load_masks(args, fold):
    mp = _masks_path(args, fold)
    if os.path.exists(mp):
        return torch.load(mp, weights_only=False)
    pp = _preds_path(args, fold)
    if os.path.exists(pp):
        d = torch.load(pp, weights_only=False)
        return {k: d[k] for k in ('train_mask', 'val_mask', 'test_mask')}
    raise FileNotFoundError(f"No masks found for fold {fold}: tried {mp} and {pp}")


def _load_conviction(args, fold):
    path = _norms_path(args, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Norms not found: {path}")
    nd = torch.load(path, weights_only=False)
    return nd[max(nd.keys())].cpu()

# ── sampling and evaluation ───────────────────────────────────────────────────

def _degree_matched_sample(hi_idx, all_deg, n, rng):
    pool = set(range(n)) - set(hi_idx.tolist())
    by_deg = collections.defaultdict(list)
    for v in pool:
        by_deg[int(all_deg[v].item())].append(v)
    for bucket in by_deg.values():
        rng.shuffle(bucket)
    used, result = set(), []
    for u in hi_idx.tolist():
        d_u   = int(all_deg[u].item())
        avail = [v for v in by_deg.get(d_u, []) if v not in used]
        if avail:
            chosen = avail[0]
        else:
            best, best_diff = None, float('inf')
            for dv, bucket in by_deg.items():
                a2 = [v for v in bucket if v not in used]
                if abs(dv - d_u) < best_diff and a2:
                    best_diff, best = abs(dv - d_u), a2[0]
            chosen = best
        used.add(chosen); result.append(chosen)
    return torch.tensor(result, dtype=torch.long)


def _mask_and_eval(model, data, mask_idx, device, is_auc=False):
    x = data.x.clone()
    if len(mask_idx) > 0:
        x[mask_idx.to(device)] = 0.0
    model.eval()
    with torch.no_grad():
        logits = model(x)
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    keep     = ~torch.isin(test_idx, mask_idx.to(device))
    eval_idx = test_idx[keep]
    if len(eval_idx) == 0:
        return float('nan'), float('nan'), float('nan')
    probs    = F.softmax(logits[eval_idx], dim=1).cpu()
    pred     = probs.argmax(1)
    y_true   = data.y[eval_idx].cpu()
    acc      = (pred == y_true).float().mean().item()
    conf     = probs.max(1).values.mean().item()
    if is_auc:
        try:
            auc = roc_auc_score(y_true.numpy(), probs[:, 1].numpy())
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')
    return acc, conf, auc

# ── per-fold experiment ───────────────────────────────────────────────────────

def _run_fold(args, dataset, model_cls, fold, device, rng):
    data   = dataset[0].clone()
    masks  = _load_masks(args, fold)
    data.train_mask = masks['train_mask']
    data.val_mask   = masks['val_mask']
    data.test_mask  = masks['test_mask']
    data   = data.to(device)
    n      = data.num_nodes

    wp = _weights_path(args, fold)
    if not os.path.exists(wp):
        print(f"  fold {fold}: weights not found — skipping"); return None
    model = model_cls(data.edge_index, args).to(device)
    model.save_other_laplacian = False
    model.load_state_dict(torch.load(wp, map_location=device, weights_only=False))
    model.eval()

    conviction   = _load_conviction(args, fold)
    deg          = pyg_degree(data.edge_index[0].cpu(), num_nodes=n).long()
    rank_conv_hi = torch.argsort(conviction, descending=True)
    rank_conv_lo = torch.argsort(conviction, descending=False)
    rank_deg_hi  = torch.argsort(deg,        descending=True)
    is_auc       = args.get('dataset') in AUC_DATASETS

    acc0, conf0, auc0 = _mask_and_eval(model, data, torch.tensor([], dtype=torch.long), device, is_auc)
    fold_result = {'baseline': {'acc': acc0, 'conf': conf0, 'auc': auc0}}
    for s in STRATEGIES:
        fold_result[s] = {'acc': [], 'conf': [], 'auc': []}

    for frac in MASK_LEVELS:
        k = max(1, int(frac * n))
        for strat, idx in [
            ('high',        rank_conv_hi[:k]),
            ('low',         rank_conv_lo[:k]),
            ('deg_high',    rank_deg_hi[:k]),
            ('deg_matched', _degree_matched_sample(rank_conv_hi[:k], deg, n, rng)),
            ('random',      torch.tensor(rng.choice(n, size=k, replace=False), dtype=torch.long)),
        ]:
            a, c, u = _mask_and_eval(model, data, idx, device, is_auc)
            fold_result[strat]['acc'].append(a)
            fold_result[strat]['conf'].append(c)
            fold_result[strat]['auc'].append(u)

    print(f"  fold {fold} done  |  baseline acc={acc0:.3f}  conf={conf0:.3f}"
          + (f"  auc={auc0:.3f}" if is_auc else ""))
    return fold_result

# ── plot ──────────────────────────────────────────────────────────────────────

def _plot(all_results, args, out_path):
    x_labels = [f"{int(p*100)}%" for p in MASK_LEVELS]
    x_pos    = list(range(len(MASK_LEVELS)))
    n_folds  = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    is_auc   = args.get('dataset') in AUC_DATASETS

    for ax, (metric, title) in zip(axes, [
        ('auc' if is_auc else 'acc',
         'ROC-AUC on unmasked test nodes' if is_auc else 'Accuracy on unmasked test nodes'),
        ('conf', 'Mean confidence on unmasked test nodes'),
    ]):
        base_mu = np.nanmean([r['baseline'].get(metric, float('nan')) for r in all_results])
        ax.axhline(base_mu, color='black', linestyle='--', linewidth=0.9,
                   label=f'No masking ({base_mu:.3f})')
        for strat, lbl, col, ls, mk in zip(STRATEGIES, LABELS, COLORS, LINESTYLES, MARKERS):
            mat = np.array([r[strat][metric] for r in all_results])
            mu  = np.nanmean(mat, 0)
            se  = np.nanstd(mat,  0) / np.sqrt(n_folds)
            ax.plot(x_pos, mu, label=lbl, color=col, linestyle=ls,
                    marker=mk, linewidth=1.8, markersize=5)
            ax.fill_between(x_pos, mu-se, mu+se, alpha=0.12, color=col)
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("Fraction of nodes masked  (X ← 0)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)

    tag = _map_tag(args)
    fig.suptitle(
        f"Conviction masking — {args['dataset']}  "
        f"({args['model']}{tag}, d={args['d']}, {args['layers']} layers)\n"
        f"X_L conviction  |  feature masking  |  unmasked test nodes  |  "
        f"mean ± SE, {n_folds} folds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    plt.close(fig)

# ── experiment discovery ──────────────────────────────────────────────────────

def _args_from_path(best_epoch_dir, model_name, normalised_str):
    epochs_dir   = os.path.dirname(os.path.dirname(best_epoch_dir))
    h_dir        = os.path.dirname(epochs_dir)
    l_dir        = os.path.dirname(h_dir)
    d_dir        = os.path.dirname(l_dir)
    dataset_dir  = os.path.dirname(os.path.dirname(os.path.dirname(d_dir)))
    try:
        epochs          = int(os.path.basename(epochs_dir).split('-')[0])
        hidden_channels = int(os.path.basename(h_dir).split('-')[0])
        layers          = int(os.path.basename(l_dir).split('-')[0])
        d               = int(os.path.basename(d_dir).split('-')[-1])
        dataset_name    = os.path.basename(dataset_dir)
    except (ValueError, IndexError):
        return None
    pth_files  = sorted(glob.glob(os.path.join(best_epoch_dir, '*.pth')))
    seed_match = re.search(r'_seed(\d+)\.pth$', pth_files[0]) if pth_files else None
    seed       = int(seed_match.group(1)) if seed_match else 43
    defaults   = vars(get_parser().parse_args([]))
    defaults.update({
        'dataset': dataset_name, 'model': model_name, 'd': d, 'layers': layers,
        'hidden_channels': hidden_channels, 'epochs': epochs, 'seed': seed,
        'normalised': normalised_str == 'true', 'deg_normalised': False,
        'folds': len(pth_files), 'edge_weights': False,
    })
    return defaults


def _discover(model_name, normalised_str, map_tag):
    weights_root = os.path.join(ROOT, 'results', 'model_weights')
    pattern = os.path.join(weights_root, '*', model_name + map_tag,
                           f'normalised-{normalised_str}',
                           'stalk_dim-*', '*-layers', '*-hidden', '*-epochs',
                           'checkpoints', 'best-epoch')
    experiments = []
    for best_epoch_dir in sorted(glob.glob(pattern)):
        pth_files = glob.glob(os.path.join(best_epoch_dir, '*.pth'))
        if not pth_files:
            continue
        epochs_dir = os.path.dirname(os.path.dirname(best_epoch_dir))
        args_path  = os.path.join(epochs_dir, 'training_args.json')
        if os.path.exists(args_path):
            with open(args_path) as f:
                targs = json.load(f)
        else:
            targs = _args_from_path(best_epoch_dir, model_name, normalised_str)
            if targs is None:
                print(f"  Could not parse path: {best_epoch_dir} — skipping"); continue
            saveable = {k: v for k, v in targs.items()
                        if isinstance(v, (str, int, float, bool, list, type(None)))}
            os.makedirs(epochs_dir, exist_ok=True)
            with open(args_path, 'w') as f:
                json.dump(saveable, f, indent=2)
            print(f"  [WARNING] Inferred training_args.json written to {args_path}")
        if targs.get('dataset', '') in EXCLUDE:
            continue
        targs['_best_epoch_dir'] = best_epoch_dir
        targs['_n_folds_found']  = len(pth_files)
        experiments.append(targs)
        print(f"  found: {targs['dataset']}  d={targs['d']}  L={targs['layers']}  folds={len(pth_files)}")
    return experiments

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--normalised', required=True, choices=['true', 'false'])
    parser.add_argument('--map_type',   default='identity',
                        help='For Joint models: identity | general | diag | bundle')
    parser.add_argument('--cuda',       type=int, default=0)
    cli = parser.parse_args()

    device = torch.device(f'cuda:{cli.cuda}' if torch.cuda.is_available() else 'cpu')

    if cli.model in _JOINT_MODELS:
        map_tag = '_first_maps_identity' if cli.map_type == 'identity' \
                  else f"_first_maps_{_TYPE_SHORT.get(cli.map_type, cli.map_type)}"
    else:
        map_tag = ''

    out_dir = os.path.join(ROOT, 'quick_analysis', 'accuracy_ranking',
                           cli.model, f"normalised_{cli.normalised}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Discovering: model={cli.model}{map_tag}  normalised={cli.normalised}")
    experiments = _discover(cli.model, cli.normalised, map_tag)
    if not experiments:
        print("No experiments found. Have you run training with --save_norms=True?"); sys.exit(1)

    print(f"\nFound {len(experiments)} experiment(s).")
    rng = np.random.default_rng(seed=42)
    MODEL_CLASSES = _model_classes()
    from utils.heterophilic import get_dataset

    for targs in experiments:
        tag  = _map_tag(targs)
        Norm = 'True' if targs.get('normalised') else 'False'
        n_folds = targs.get('folds', targs['_n_folds_found'])
        print(f"\n{'='*60}\n  {targs['dataset']}  |  {targs['model']}{tag}  "
              f"d={targs['d']}  L={targs['layers']}  folds={n_folds}\n{'='*60}")

        if targs['model'] not in MODEL_CLASSES:
            print(f"  Unknown model '{targs['model']}' — skipping"); continue

        class _FakeArgs:
            pass
        fake = _FakeArgs()
        for k, v in targs.items(): setattr(fake, k, v)
        try:
            dataset = get_dataset(targs['dataset'], fake)
        except Exception as e:
            print(f"  Could not load dataset: {e} — skipping"); continue

        adict = dict(targs)
        adict['device']      = device
        adict['input_dim']   = dataset[0].x.shape[1]
        adict['graph_size']  = dataset[0].x.size(0)
        try:    adict['output_dim'] = dataset.num_classes
        except: adict['output_dim'] = int(torch.unique(dataset[0].y).shape[0])
        if adict.get('sheaf_decay') is None:
            adict['sheaf_decay'] = adict['weight_decay']

        model_cls   = MODEL_CLASSES[targs['model']]
        all_results = []
        for fold in range(n_folds):
            result = _run_fold(adict, dataset, model_cls, fold, device, rng)
            if result is not None:
                all_results.append(result)

        if not all_results:
            print("  No folds completed — skipping plot"); continue

        out_path = os.path.join(out_dir,
            f"{targs['dataset']}_{targs['model']}{tag}_d{targs['d']}_{targs['layers']}L_"
            f"Norm_{Norm}_conviction_masking.pdf")
        _plot(all_results, adict, out_path)

    print(f"\nDone. Plots in: {out_dir}")

"""
Shared conviction utilities: locating a fold's scoring checkpoint, computing its
conviction ranking, and applying the three Phase-2 interventions (node pruning,
feature masking, message gating) to a Data object.

Used by quick_analysis/conviction_masking_forward.py (inference-only diagnostic,
covering masking/pruning/gating) and run_conviction_experiments.py
(retrain-from-scratch pipeline), so both scripts locate checkpoints and rank
nodes the exact same way.
"""

import os
import re
import glob
import json
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data

from definitions import ROOT_DIR
from exp.parser import get_parser

# ── constants ─────────────────────────────────────────────────────────────────

JOINT_MODELS = {'JointSheafParams', 'JointSheafParamsAlt'}
TYPE_SHORT   = {'diagonal': 'diag', 'bundle': 'bundle', 'general': 'general'}
EXCLUDE      = {'synthetic_exp'}


@dataclass
class PruneConfig:
    type: str    # 'node' | 'feature' | 'message'
    pct: float
    mode: str    # 'zero'/'mean' (feature) | 'hard'/'soft' (message) | unused (node)
    score: str   # 'last' | 'avg' | 'avg_z'

# ── path helpers ──────────────────────────────────────────────────────────────

def map_tag(args):
    if args.get('model') not in JOINT_MODELS:
        return ""
    if not args.get('learn_first_maps', False):
        return "_first_maps_identity"
    short = TYPE_SHORT.get(args.get('learnt_map_type', 'general'), 'general')
    return f"_first_maps_{short}"


def base_dir(result_type, args):
    return os.path.join(
        ROOT_DIR, 'results', result_type,
        args['dataset'],
        args['model'] + map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    )


def pt_filename(args, fold):
    return f"{args['model']}_{args['dataset']}{map_tag(args)}_fold{fold}_seed{args['seed']}.pt"


def weights_path(args, fold):
    fn = pt_filename(args, fold).replace('.pt', '.pth')
    return os.path.join(base_dir('model_weights', args), 'checkpoints', 'best-epoch', fn)


def masks_path(args, fold):
    fn = pt_filename(args, fold).replace('.pt', '_masks.pt')
    return os.path.join(base_dir('model_weights', args), 'checkpoints', 'best-epoch', fn)


def preds_path(args, fold):
    return os.path.join(base_dir('node_preds', args), pt_filename(args, fold))


def norms_path(args, fold):
    return os.path.join(base_dir('node_norms', args), 'checkpoints', 'best-epoch',
                        pt_filename(args, fold))


def confidence_path(args, fold):
    return os.path.join(base_dir('node_conf', args), 'checkpoints', 'best-epoch',
                        pt_filename(args, fold))

# ── loading ───────────────────────────────────────────────────────────────────

def load_masks(args, fold):
    mp = masks_path(args, fold)
    if os.path.exists(mp):
        return torch.load(mp, weights_only=False)
    pp = preds_path(args, fold)
    if os.path.exists(pp):
        d = torch.load(pp, weights_only=False)
        return {k: d[k] for k in ('train_mask', 'val_mask', 'test_mask')}
    raise FileNotFoundError(f"No masks found for fold {fold}: tried {mp} and {pp}")


def load_conviction(args, fold, score=None):
    score = score or args.get('conviction_score', 'last')
    path = norms_path(args, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Norms not found: {path}")
    nd = torch.load(path, weights_only=False)
    if score in ('avg', 'avg_z'):
        post_diffusion_layers = [k for k in nd.keys() if k > 0]
        layer_norms = [nd[k] for k in post_diffusion_layers]
        if score == 'avg_z':
            layer_norms = [(v - v.mean()) / v.std().clamp_min(1e-8) for v in layer_norms]
        return torch.stack(layer_norms).mean(dim=0).cpu()
    return nd[max(nd.keys())].cpu()


def load_confidence(args, fold):
    """Max-softmax prediction confidence per node, from the same best-epoch
    (validation-selected) checkpoint as load_conviction."""
    path = confidence_path(args, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Confidence not found: {path}")
    return torch.load(path, weights_only=False).cpu()

# ── ranking ───────────────────────────────────────────────────────────────────

def _pool_idx(masks, pool):
    if pool == 'train':
        pool_mask = masks['train_mask']
    elif pool == 'val':
        pool_mask = masks['val_mask']
    elif pool == 'trainval':
        pool_mask = masks['train_mask'] | masks['val_mask']
    else:
        raise ValueError(f"Unknown pool: {pool}")
    return pool_mask.nonzero(as_tuple=True)[0].cpu()


def conviction_ranking(args, fold, score='last', pool='train'):
    """Full sorted (ascending, lowest conviction first) node-id ranking over `pool`.
    Caller slices ranking[:k] for whatever budget k it needs."""
    idx        = _pool_idx(load_masks(args, fold), pool)
    conviction = load_conviction(args, fold, score=score)
    return idx[torch.argsort(conviction[idx], descending=False)]


def confidence_ranking(args, fold, pool='train'):
    """Full sorted (ascending, lowest prediction-confidence first) node-id
    ranking over `pool` — Alessio's "low prediction-confidence pruning" baseline."""
    idx        = _pool_idx(load_masks(args, fold), pool)
    confidence = load_confidence(args, fold)
    return idx[torch.argsort(confidence[idx], descending=False)]

# ── interventions ─────────────────────────────────────────────────────────────

def _class_balance(y, mask):
    """Per-class fraction of y[mask]. Empty dict if the mask selects no nodes."""
    y_sub = y[mask].cpu().numpy()
    if len(y_sub) == 0:
        return {}
    classes, counts = np.unique(y_sub, return_counts=True)
    return {int(c): float(cnt) / len(y_sub) for c, cnt in zip(classes, counts)}


def prune_nodes(data, remove_ids):
    """Remove remove_ids + incident edges, relabel to a new contiguous id space.
    Returns (new_data, diagnostics) — diagnostics covers Alessio's required table:
    pct nodes/edges remaining, # connected components, class balance (per split),
    train/val/test split composition before vs. after, giant component size."""
    n = data.num_nodes
    remove_ids = remove_ids.to(data.x.device)
    assert not (data.val_mask[remove_ids] | data.test_mask[remove_ids]).any(), \
        "prune_nodes: remove_ids must not include val/test nodes"
    keep_mask = torch.ones(n, dtype=torch.bool, device=data.x.device)
    keep_mask[remove_ids] = False
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]

    new_id = torch.full((n,), -1, dtype=torch.long, device=data.x.device)
    new_id[keep_idx] = torch.arange(keep_idx.numel(), device=data.x.device)

    src, dst = data.edge_index[0], data.edge_index[1]
    edge_keep = keep_mask[src] & keep_mask[dst]
    new_edge_index = torch.stack([new_id[src[edge_keep]], new_id[dst[edge_keep]]], dim=0)

    new_data = Data(x=data.x[keep_idx], y=data.y[keep_idx], edge_index=new_edge_index)
    new_data.train_mask = data.train_mask[keep_idx]
    new_data.val_mask   = data.val_mask[keep_idx]
    new_data.test_mask  = data.test_mask[keep_idx]

    n_new, m_old, m_new = keep_idx.numel(), src.numel(), new_edge_index.size(1)
    adj = coo_matrix(
        (np.ones(m_new), (new_edge_index[0].cpu().numpy(), new_edge_index[1].cpu().numpy())),
        shape=(n_new, n_new),
    )
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes  = np.bincount(labels) if n_new > 0 else np.array([0])
    giant_size  = int(comp_sizes.max())

    class_balance = {
        'train': _class_balance(new_data.y, new_data.train_mask),
        'val':   _class_balance(new_data.y, new_data.val_mask),
        'test':  _class_balance(new_data.y, new_data.test_mask),
    }

    # split composition drifts even though val/test counts are untouched, since
    # only train shrinks: e.g. an original 60/20/20 split becomes something else
    # as a fraction of the smaller pruned graph.
    split_pct_before = {
        'train': float(data.train_mask.sum()) / n,
        'val':   float(data.val_mask.sum())   / n,
        'test':  float(data.test_mask.sum())  / n,
    }
    split_pct_after = {
        'train': float(new_data.train_mask.sum()) / n_new if n_new > 0 else 0.0,
        'val':   float(new_data.val_mask.sum())   / n_new if n_new > 0 else 0.0,
        'test':  float(new_data.test_mask.sum())  / n_new if n_new > 0 else 0.0,
    }

    diagnostics = {
        'pct_nodes_remaining':    n_new / n,
        'pct_edges_remaining':    m_new / m_old if m_old > 0 else float('nan'),
        'n_connected_components': int(n_components),
        'class_balance':          class_balance,
        'split_pct_before':       split_pct_before,
        'split_pct_after':        split_pct_after,
        'giant_component_size':   giant_size,
        'giant_component_pct':    giant_size / n_new if n_new > 0 else 0.0,
    }
    return new_data, diagnostics


def mask_features(data, mask_ids, mode='zero'):
    """Zero or mean-replace features of mask_ids. Topology/edges untouched."""
    new_data = data.clone()
    mask_ids = mask_ids.to(new_data.x.device)
    if len(mask_ids) == 0:
        return new_data
    if mode == 'zero':
        new_data.x[mask_ids] = 0.0
    elif mode == 'mean':
        new_data.x[mask_ids] = data.x.mean(dim=0)
    else:
        raise ValueError(f"Unknown feature mask mode: {mode}")
    return new_data


def build_message_gate(n_nodes, conviction, gate_ids, mode='hard', min_scale=0.1, device=None):
    """Per-node outgoing-message multiplier: 1.0 = full message (default for all
    nodes not in gate_ids). mode='hard' zeroes gate_ids' messages; mode='soft'
    attenuates them proportionally to their conviction (lowest -> min_scale,
    highest of the gated set -> ~1.0)."""
    gate = torch.ones(n_nodes, dtype=torch.float, device=device)
    if len(gate_ids) == 0:
        return gate
    gate_ids = gate_ids.to(gate.device)
    if mode == 'hard':
        gate[gate_ids] = 0.0
    elif mode == 'soft':
        vals = conviction[gate_ids.cpu()].to(gate.device)
        span = (vals.max() - vals.min()).clamp_min(1e-8)
        gate[gate_ids] = min_scale + (1 - min_scale) * (vals - vals.min()) / span
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    return gate

# ── model registry ────────────────────────────────────────────────────────────

def model_classes():
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

# ── experiment discovery ──────────────────────────────────────────────────────

def infer_args_from_path(best_epoch_dir, model_name, normalised_str):
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


def discover_experiments(model_name, normalised_str, tag):
    weights_root = os.path.join(ROOT_DIR, 'results', 'model_weights')
    pattern = os.path.join(weights_root, '*', model_name + tag,
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
            targs = infer_args_from_path(best_epoch_dir, model_name, normalised_str)
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

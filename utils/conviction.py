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
import collections
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import degree as pyg_degree

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

# ── candidate selection (shared by the forward-eval and retrain scripts) ───────

def load_fold_rankings(args, fold, edge_index, num_nodes, score='last', pool='train'):
    """Bundle of everything select_candidates needs for one fold: the candidate
    pool, per-node degree, and the deterministic strategy rankings (conviction
    hi/lo, degree hi/lo, confidence lo). Computing this once per fold and handing
    the same dict to select_candidates is what guarantees the forward-eval and
    retrain scripts pick identical candidates for the same (strategy, budget, fold)."""
    pool_idx     = _pool_idx(load_masks(args, fold), pool)
    deg          = pyg_degree(edge_index[0].cpu(), num_nodes=num_nodes).long()
    deg_pool     = deg[pool_idx]
    rank_conv_lo = conviction_ranking(args, fold, score=score, pool=pool)
    rank_conv_hi = torch.flip(rank_conv_lo, dims=[0])
    rank_deg_hi  = pool_idx[torch.argsort(deg_pool, descending=True)]
    rank_deg_lo  = pool_idx[torch.argsort(deg_pool, descending=False)]
    rank_conf_lo = confidence_ranking(args, fold, pool=pool)
    return {
        'pool_idx': pool_idx, 'deg': deg,
        'rank_conv_lo': rank_conv_lo, 'rank_conv_hi': rank_conv_hi,
        'rank_deg_hi': rank_deg_hi, 'rank_deg_lo': rank_deg_lo,
        'rank_conf_lo': rank_conf_lo,
    }


def _degree_matched_sample(hi_idx, all_deg, pool_idx, rng):
    """k random nodes from the pool (excluding hi_idx) whose degree distribution
    matches hi_idx's, one draw per fold. Greedy nearest-degree matching without
    replacement; falls back to the closest available degree bucket if a node's
    exact degree is exhausted."""
    pool = set(pool_idx.tolist()) - set(hi_idx.tolist())
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


def select_candidates(strategy, k, *, pool_idx, deg, rank_conv_lo, rank_conv_hi,
                       rank_deg_hi, rank_deg_lo, rank_conf_lo, rng):
    """Return the k candidate node ids for one of the 8 canonical strategies
    (high/low conviction, high/low degree, degree-matched high/low, low
    confidence, random). Shared dispatch so the forward-eval and retrain scripts
    draw identical candidates for the same (strategy, k, fold) — note 'random'
    and 'deg_matched_*' consume rng, so they only match across scripts if called
    in the same order with the same seed."""
    if strategy == 'high':
        return rank_conv_hi[:k]
    if strategy == 'low':
        return rank_conv_lo[:k]
    if strategy == 'deg_high':
        return rank_deg_hi[:k]
    if strategy == 'deg_low':
        return rank_deg_lo[:k]
    if strategy == 'deg_matched_high':
        return _degree_matched_sample(rank_conv_hi[:k], deg, pool_idx, rng)
    if strategy == 'deg_matched_low':
        return _degree_matched_sample(rank_conv_lo[:k], deg, pool_idx, rng)
    if strategy == 'conf_low':
        return rank_conf_lo[:k]
    if strategy == 'random':
        return torch.tensor(rng.choice(pool_idx.numpy(), size=k, replace=False), dtype=torch.long)
    raise ValueError(f"Unknown strategy: {strategy}")

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

# ── evaluation (shared by the forward-eval and retrain scripts) ────────────────

def forward_eval(model, data, eval_idx, device, is_auc=False, gate=None):
    """Forward pass + metrics on eval_idx. model must already match data's
    node count/edge_index (relevant after pruning, where a fresh model is built
    for the smaller graph). gate is the optional per-node outgoing-message
    multiplier (see _apply_gate in disc_models.py); None (default) is a
    complete no-op for every model."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, gate=gate)
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


def evaluate_checkpoint(args, fold, model_cls, data, device, is_auc=False):
    """Load the frozen best-epoch checkpoint and forward-eval it on the
    unperturbed test set. This is the 'reference' number for the retrain
    experiments (run_conviction_experiments.py): what the model that's
    actually trained and saved got, as opposed to a fresh retrain's baseline.
    One forward pass, not a retrain — effectively free."""
    wp = weights_path(args, fold)
    if not os.path.exists(wp):
        raise FileNotFoundError(f"No checkpoint found for fold {fold}: {wp}")
    model = model_cls(data.edge_index, args).to(device)
    state_dict = torch.load(wp, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    acc, conf, auc = forward_eval(model, data, test_idx, device, is_auc)
    return {'acc': acc, 'conf': conf, 'auc': auc}

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


def discover_experiments(model_name, normalised_str, tag, dataset_filter=None):
    """dataset_filter restricts discovery to one dataset directory instead of
    globbing '*' — lets a single SLURM job own exactly one dataset (needed for
    the retrain sweep, where each discovered experiment is far more expensive
    to process than in the forward-eval script)."""
    weights_root = os.path.join(ROOT_DIR, 'results', 'model_weights')
    pattern = os.path.join(weights_root, dataset_filter or '*', model_name + tag,
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

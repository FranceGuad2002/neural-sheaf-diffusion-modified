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
import csv
import glob
import json
import zlib
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

# ── constants ─────────────────────────────────────────────────────────────────

JOINT_MODELS = {'JointSheafParams', 'JointSheafParamsAlt'}
TYPE_SHORT   = {'diagonal': 'diag', 'bundle': 'bundle', 'general': 'general'}
EXCLUDE      = {'synthetic_exp'}

# Binary, heavily class-imbalanced datasets scored by ROC-AUC rather than
# accuracy. Single source of truth for every conviction analysis script, so a
# dataset can never be tabulated under the wrong metric (exp/run.py keeps its
# own copy for the training loop; the two must agree).
AUC_DATASETS = {'minesweeper', 'tolokers', 'questions'}


def metric_key(dataset):
    """The headline metric for a dataset: 'auc' for the ROC-AUC datasets,
    'acc' everywhere else. Use this instead of testing AUC_DATASETS inline, so
    plots and tables can never disagree about which column is the result."""
    return 'auc' if dataset in AUC_DATASETS else 'acc'


def metric_label(dataset):
    return 'ROC-AUC' if dataset in AUC_DATASETS else 'Accuracy'


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
    Caller slices ranking[:k] for whatever budget k it needs. Ties fall back to
    ascending node id; load_fold_rankings does NOT use this — it applies a random
    tie-break instead (see _argsort_with_tiebreak) for permutation invariance."""
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

def draw_rng(dataset, fold, strategy, frac, base_seed=42):
    """Generator for ONE candidate draw, derived from the identity of that draw
    rather than shared as a running stream.

    The randomised strategies ('random', 'deg_matched_*') used to consume a
    single module-level generator threaded through every dataset, fold, budget
    and strategy. Because a generator is a stream, each draw then depended on
    how many draws preceded it — so running a subset of the manifest, reordering
    it, or using a different strategy list (as the retrain script does) silently
    changed the candidates for every dataset after the first.

    Keying on (dataset, fold, strategy, frac) makes each draw independent and
    reproducible: the same key always yields the same nodes, whatever else ran
    before. It also makes the forward-eval and retrain scripts agree on the
    randomised arms despite sweeping different strategy/budget lists.

    crc32, not the builtin hash(): Python salts string hashing per process
    (PYTHONHASHSEED), which would reintroduce exactly the irreproducibility
    this function exists to remove — and only across runs, the hardest kind
    to notice."""
    key = zlib.crc32(f"{dataset}|{strategy}|{frac}".encode()) & 0xFFFFFFFF
    return np.random.default_rng([int(base_seed), int(key), int(fold)])


def _fold_tiebreak(num_nodes, args, fold, tiebreak_seed=None):
    """Per-node random tie-break key (float in [0,1), length num_nodes, indexed
    by global node id) for one fold. Seeded deterministically from args['seed']
    and fold so the forward-eval and retrain scripts derive the SAME key and
    therefore still pick identical candidates for a given (strategy, budget,
    fold). Drawn from its own generator, independent of the sampling rng used by
    the 'random'/'deg_matched' strategies, so those streams are unchanged."""
    if tiebreak_seed is None:
        tiebreak_seed = (int(args['seed']) * 1_000_003 + int(fold) * 9_973 + 0x5EED) & 0xFFFFFFFF
    return np.random.default_rng(tiebreak_seed).random(num_nodes)


def _argsort_with_tiebreak(values, idx, tiebreak, descending):
    """Return `idx` reordered by `values[idx]`, breaking exact ties by
    `tiebreak[idx]` (a per-node random key) instead of by ascending node id.
    This removes the default low-id preference among equal-valued nodes, so
    which of several tied nodes fall inside a top-k / bottom-k budget no longer
    depends on the arbitrary graph labelling — the selection is permutation
    invariant. `tiebreak` is indexed by global node id (same space as `idx`)."""
    idx_np      = idx.cpu().numpy()
    primary     = values[idx].cpu().numpy().astype(np.float64)
    primary_key = -primary if descending else primary
    # np.lexsort's LAST key is the primary sort key; `tiebreak` resolves ties.
    order = np.lexsort((np.asarray(tiebreak)[idx_np], primary_key))
    return idx[torch.as_tensor(order, dtype=torch.long)]


def load_fold_rankings(args, fold, edge_index, num_nodes, score='last', pool='train',
                       tiebreak_seed=None):
    """Bundle of everything select_candidates needs for one fold: the candidate
    pool, per-node degree, and the strategy rankings (conviction hi/lo, degree
    hi/lo, confidence hi/lo). Computing this once per fold and handing the same
    dict to select_candidates is what guarantees the forward-eval and retrain
    scripts pick identical candidates for the same (strategy, budget, fold).

    Ties in each ranking (very common for degree) are broken by a per-node
    random key (see _fold_tiebreak) rather than by node id, so selection does
    not systematically favour low-id nodes and is permutation invariant. Both
    scripts derive the same key from args['seed']+fold, preserving cross-script
    agreement; pass tiebreak_seed to override."""
    pool_idx     = _pool_idx(load_masks(args, fold), pool)
    deg          = pyg_degree(edge_index[0].cpu(), num_nodes=num_nodes).long()
    tiebreak     = _fold_tiebreak(num_nodes, args, fold, tiebreak_seed)
    conviction   = load_conviction(args, fold, score=score)
    confidence   = load_confidence(args, fold)
    rank_conv_lo = _argsort_with_tiebreak(conviction, pool_idx, tiebreak, descending=False)
    rank_conv_hi = _argsort_with_tiebreak(conviction, pool_idx, tiebreak, descending=True)
    rank_deg_hi  = _argsort_with_tiebreak(deg,        pool_idx, tiebreak, descending=True)
    rank_deg_lo  = _argsort_with_tiebreak(deg,        pool_idx, tiebreak, descending=False)
    rank_conf_lo = _argsort_with_tiebreak(confidence, pool_idx, tiebreak, descending=False)
    rank_conf_hi = _argsort_with_tiebreak(confidence, pool_idx, tiebreak, descending=True)
    return {
        'pool_idx': pool_idx, 'deg': deg,
        'rank_conv_lo': rank_conv_lo, 'rank_conv_hi': rank_conv_hi,
        'rank_deg_hi': rank_deg_hi, 'rank_deg_lo': rank_deg_lo,
        'rank_conf_lo': rank_conf_lo, 'rank_conf_hi': rank_conf_hi,
    }


def _degree_matched_sample(hi_idx, all_deg, pool_idx, rng):
    """k random nodes from the pool (excluding hi_idx) whose degree distribution
    matches hi_idx's, one draw per fold. Greedy nearest-degree matching without
    replacement; falls back to the closest available degree bucket if a node's
    exact degree is exhausted."""
    pool = set(pool_idx.tolist()) - set(hi_idx.tolist())
    # Sampling k nodes without replacement from pool\hi_idx needs
    # k <= |pool| - k. Budgets up to 30% satisfy this comfortably; beyond 50%
    # the greedy loop would run out of candidates and append None, which only
    # surfaces as an opaque TypeError in torch.tensor() below.
    if len(hi_idx) > len(pool):
        raise ValueError(
            f"degree-matched sampling needs a budget <= 50% of the pool: asked for "
            f"{len(hi_idx)} nodes but only {len(pool)} remain after excluding the "
            f"reference set")
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
                       rank_deg_hi, rank_deg_lo, rank_conf_lo, rank_conf_hi, rng):
    """Return the k candidate node ids for one of the 9 canonical strategies
    (high/low conviction, high/low degree, degree-matched high/low, high/low
    confidence, random). Shared dispatch so the forward-eval and retrain scripts
    draw identical candidates for the same (strategy, k, fold).

    That guarantee holds for the randomised strategies ('random',
    'deg_matched_*') only when `rng` is built per draw with draw_rng() — pass a
    shared running generator instead and the two scripts diverge, since they
    sweep different strategy/budget lists and so consume it at different rates."""
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
    if strategy == 'conf_high':
        return rank_conf_hi[:k]
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

# ── table output ──────────────────────────────────────────────────────────────

TABLES_ROOT_NAME = 'Conviction_experiments_tables'
PLOTS_ROOT_NAME  = 'Conviction_experiments'


def tables_path(plot_path, ext='.csv'):
    """Mirror a plot path from the Conviction_experiments/ tree into the
    parallel Conviction_experiments_tables/ tree, swapping the extension.
    Directory layout below the root is identical in both, so a table always
    sits at the same relative location as the figure it tabulates."""
    parts = os.path.normpath(plot_path).split(os.sep)
    try:
        i = parts.index(PLOTS_ROOT_NAME)
    except ValueError:
        raise ValueError(f"'{PLOTS_ROOT_NAME}' not in path: {plot_path}")
    parts[i] = TABLES_ROOT_NAME
    return os.path.splitext(os.sep.join(parts))[0] + ext


def table_meta(args, intervention, mode=''):
    """Identifying columns repeated on every row, so tables from different
    datasets/models can be concatenated and pivoted without losing provenance."""
    return {
        'dataset':          args['dataset'],
        'model':            args['model'] + map_tag(args),
        'd':                args['d'],
        'layers':           args['layers'],
        'normalised':       str(args.get('normalised')).lower(),
        'conviction_score': args.get('conviction_score', 'last'),
        'intervention':     intervention,
        'mode':             mode,
        'metric':           metric_key(args['dataset']),
    }


META_FIELDS = ['dataset', 'model', 'd', 'layers', 'normalised', 'conviction_score',
               'intervention', 'mode', 'metric']


def write_table(rows, out_path, extra_fields):
    """Write rows (list of dicts) as CSV under the tables tree. Columns are the
    shared meta fields followed by extra_fields."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS + extra_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved table to {out_path}")


def mean_se(values, n_folds):
    """Fold-mean and standard error, NaN-tolerant (a fold whose metric is NaN —
    e.g. AUC on a degenerate split — is skipped rather than poisoning the mean)."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr) / np.sqrt(n_folds))

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

# ── manifest-based experiment loading (no glob discovery) ──────────────────────

def load_manifest(manifest_path=None):
    """Parse the plain-text experiment manifest into a list of dicts. Each
    non-comment line: dataset model normalised d layers hidden epochs
    (whitespace-separated); '#' starts a comment, blank lines ignored.
    Replaces discover_experiments' glob walk over results/model_weights/ — the
    set of trained models is small and known, so it's edited by hand instead
    of rediscovered on every run."""
    path = manifest_path or os.path.join(ROOT_DIR, 'quick_analysis', 'conviction_manifest.txt')
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            dataset, model, normalised, d, layers, hidden, epochs = line.split()
            rows.append({
                'dataset': dataset, 'model': model, 'normalised': normalised.lower(),
                'd': int(d), 'layers': int(layers),
                'hidden_channels': int(hidden), 'epochs': int(epochs),
            })
    return rows


def load_manifest_experiments(model_name, normalised_str, tag, manifest_path=None, dataset_filter=None):
    """Manifest-based replacement for discover_experiments: same return shape
    (list of training_args dicts carrying '_best_epoch_dir'/'_n_folds_found'),
    but resolves each entry's checkpoint directory directly from the manifest
    instead of globbing the whole results/ tree."""
    experiments = []
    for row in load_manifest(manifest_path):
        if row['model'] != model_name or row['normalised'] != normalised_str:
            continue
        if dataset_filter and row['dataset'] != dataset_filter:
            continue
        best_epoch_dir = os.path.join(
            ROOT_DIR, 'results', 'model_weights', row['dataset'], model_name + tag,
            f"normalised-{normalised_str}", f"stalk_dim-{row['d']}",
            f"{row['layers']}-layers", f"{row['hidden_channels']}-hidden",
            f"{row['epochs']}-epochs", 'checkpoints', 'best-epoch')
        args_path = os.path.join(os.path.dirname(os.path.dirname(best_epoch_dir)), 'training_args.json')
        if not os.path.exists(args_path):
            print(f"  [WARNING] No training_args.json for {row['dataset']} — skipping"); continue
        with open(args_path) as f:
            targs = json.load(f)
        pth_files = glob.glob(os.path.join(best_epoch_dir, '*.pth'))
        if not pth_files:
            print(f"  [WARNING] No checkpoints for {row['dataset']} — skipping"); continue
        if targs.get('dataset', '') in EXCLUDE:
            continue
        targs['_best_epoch_dir'] = best_epoch_dir
        targs['_n_folds_found']  = len(pth_files)
        experiments.append(targs)
        print(f"  loaded: {targs['dataset']}  d={targs['d']}  L={targs['layers']}  folds={len(pth_files)}")
    return experiments

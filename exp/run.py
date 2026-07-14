#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from ast import arg
import enum
from math import e
import sys
import os
import json
import random
from datetime import datetime
import torch
import pandas as pd
import torch.nn.functional as F
import git
import numpy as np
import networkx as nx
import scipy.sparse as sp
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
from models.disc_models import (DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion, DiscreteVanillaDiffusion, DiscreteVanillaDiffusionAlt,
    DiscreteJointSheafVanillaDiffusion, DiscreteJointSheafDiffusionParams,
    DiscreteJointSheafDiffusionParamsAlt)
from utils.heterophilic import get_dataset, get_fixed_splits

AUC_DATASETS = {'minesweeper', 'tolokers', 'questions'}


def precompute_sheaf_mappings(data, d, args):
    diff_model = DiscreteJointSheafVanillaDiffusion(data.edge_index, args).to(args['device'])
    U, S, V = torch.pca_lowrank(data.x)
    x_d = torch.matmul(data.x, V[:, 0:d])
    with torch.no_grad():
        sheaf_init = diff_model(x_d)
    return sheaf_init


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data, gate=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, gate=gate)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data, gate=None):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds, probs = model(data.x, gate=gate), [], [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
            probs.append(torch.softmax(logits[mask], dim = 1).detach().cpu())
        return accs, preds, losses, probs


_JOINT_MODELS = {'JointSheafParams', 'JointSheafParamsAlt'}
_TYPE_SHORT   = {'diagonal': 'diag', 'bundle': 'bundle', 'general': 'general'}


def _map_tag(args):
    """Filename suffix encoding the first-map type for Joint models; empty for all others."""
    if args['model'] not in _JOINT_MODELS:
        return ""
    if not args['learn_first_maps']:
        return "_first_maps_identity"
    short = _TYPE_SHORT.get(args['learnt_map_type'], args['learnt_map_type'])
    return f"_first_maps_{short}"


def _lap_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'laplacians',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _norms_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'node_norms',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_norms(norms_dict, args, fold, norms_dir):
    os.makedirs(norms_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}.pt"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}.pt"
    path = os.path.join(norms_dir, filename)
    torch.save(norms_dict, path)
    print(f"Saved node norms to {path}")


def _preds_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'node_preds',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_node_preds(node_preds_dict, args, fold, preds_dir):
    os.makedirs(preds_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}.pt"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}.pt"
    path = os.path.join(preds_dir, filename)
    torch.save(node_preds_dict, path)
    print(f"Saved node predictions to {path}")


def _saliency_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'node_saliency',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_saliency(saliency, args, fold, saliency_dir):
    os.makedirs(saliency_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}.pt"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}.pt"
    path = os.path.join(saliency_dir, filename)
    torch.save(saliency, path)
    print(f"Saved saliency to {path}")


def _conf_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'node_conf',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_conf(conf, args, fold, conf_dir):
    os.makedirs(conf_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}.pt"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}.pt"
    path = os.path.join(conf_dir, filename)
    torch.save(conf, path)
    print(f"Saved confidence to {path}")


def _acc_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'node_accuracy',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_acc(correct, args, fold, acc_dir):
    os.makedirs(acc_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}.pt"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}.pt"
    path = os.path.join(acc_dir, filename)
    torch.save(correct, path)
    print(f"Saved per-node accuracy to {path}")


def _weights_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'model_weights',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_weights(state_dict, args, fold, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}.pth"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}.pth"
    path = os.path.join(weights_dir, filename)
    torch.save(state_dict, path)
    print(f"Saved model weights to {path}")


def _save_training_args(args, weights_base):
    """Save JSON of training hyperparameters once per experiment (shared across folds)."""
    os.makedirs(weights_base, exist_ok=True)
    path = os.path.join(weights_base, 'training_args.json')
    if os.path.exists(path):
        return
    saveable = {k: v for k, v in args.items()
                if isinstance(v, (str, int, float, bool, list, type(None)))}
    with open(path, 'w') as f:
        json.dump(saveable, f, indent=2)
    print(f"Saved training args to {path}")


def _save_masks(data, args, fold, weights_dir):
    """Save fold-specific train/val/test masks alongside model weights."""
    os.makedirs(weights_dir, exist_ok=True)
    tag = _map_tag(args)
    if args['dataset'] == 'synthetic_exp':
        filename = (
            f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
            f"_pct-hetero-{int(float(args['het_coef'])*100)}"
            f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
            f"{tag}_fold{fold}_seed{args['seed']}_masks.pt"
        )
    else:
        filename = f"{args['model']}_{args['dataset']}{tag}_fold{fold}_seed{args['seed']}_masks.pt"
    path = os.path.join(weights_dir, filename)
    torch.save({
        'train_mask': data.train_mask.cpu(),
        'val_mask':   data.val_mask.cpu(),
        'test_mask':  data.test_mask.cpu(),
    }, path)
    print(f"Saved fold masks to {path}")


def _compute_conf_saliency(model, data):
    """One forward + one backward pass → per-node confidence and saliency (n,) each."""
    model.eval()
    all_mask = data.train_mask | data.val_mask | data.test_mask
    X_in  = data.x.detach().clone().requires_grad_(True)
    logits = model(X_in)
    loss   = F.nll_loss(logits[all_mask], data.y[all_mask])
    (grad_X,) = torch.autograd.grad(loss, X_in, create_graph=False)
    logits_d  = logits.detach()
    pred      = logits_d.max(1)[1].cpu()
    conf      = torch.softmax(logits_d, dim=1).max(1)[0].cpu()
    saliency  = grad_X.norm(dim=1).cpu()
    return pred, conf, saliency


def _save_laplacians(laplacian_dict, args, fold, lap_dir):
    os.makedirs(lap_dir, exist_ok=True)
    tag = _map_tag(args)

    for layer, lap in laplacian_dict.items():
        lap_indices = lap[0].detach().cpu()
        lap_values = lap[1].detach().cpu()

        if lap_indices.dim() != 2 or lap_indices.size(0) != 2:
            raise ValueError(
                f"Expected Laplacian indices of shape [2, N], got {tuple(lap_indices.shape)}")

        lap_matrix = torch.cat([lap_indices, lap_values.unsqueeze(0)], dim=0)

        if args['dataset'] == 'synthetic_exp':
            lap_filename = (
                f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}"
                f"_layer{layer}_pct-hetero-{int(float(args['het_coef']) * 100)}"
                f"_classes-{args['num_classes']}_feats-{args['num_feats']}"
                f"{tag}_seed{args['seed']}.pt"
            )
        else:
            lap_filename = (
                f"{args['model']}_{args['dataset']}_layer{layer}"
                f"_fold{fold}{tag}_seed{args['seed']}.pt"
            )

        lap_path = os.path.join(lap_dir, lap_filename)
        torch.save(lap_matrix, lap_path)
        print(f"Saved Laplacian to {lap_path} with shape {tuple(lap_matrix.shape)}")


def _forman_eigs_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'forman_eigs',
        args['dataset'],
        args['model'] + _map_tag(args),
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _forman_eigs_other_base_dir(args):
    return os.path.join(_forman_eigs_base_dir(args), 'other_Laplacian')


def _compute_F_diags(lap_indices, lap_values, n, d):
    """Forman curvature diagonal blocks F_{u,u} from COO (indices [2,M], values [M])."""
    row_idx = lap_indices[0].numpy().astype(np.int64)
    col_idx = lap_indices[1].numpy().astype(np.int64)
    values  = lap_values.numpy()

    block_row, block_col = row_idx // d, col_idx // d
    local_r,   local_c   = row_idx  % d, col_idx  % d
    is_diag = (block_row == block_col)

    diag_blocks = np.zeros((n, d, d))
    dm = is_diag
    np.add.at(diag_blocks, (block_row[dm], local_r[dm], local_c[dm]), values[dm])

    off        = ~is_diag
    block_ids  = block_row[off] * n + block_col[off]
    unique_ids, inv = np.unique(block_ids, return_inverse=True)
    all_blocks = np.zeros((len(unique_ids), d, d))
    np.add.at(all_blocks, (inv, local_r[off], local_c[off]), values[off])

    nz = np.any(all_blocks != 0, axis=(1, 2))
    abs_blocks = np.zeros_like(all_blocks)
    if np.any(nz):
        U, S, _ = np.linalg.svd(all_blocks[nz])
        abs_blocks[nz] = np.einsum("bij,bj,bkj->bik", U, S, U)

    B_diag = np.zeros((n, d, d))
    np.add.at(B_diag, (unique_ids // n).astype(np.int64), abs_blocks)
    return diag_blocks - B_diag


def _build_f0_top(edge_index, n):
    """Topological Forman curvature from PyG edge_index (node-level indices)."""
    ei    = edge_index.cpu().numpy()
    edges = {(min(int(u), int(v)), max(int(u), int(v)))
             for u, v in zip(ei[0], ei[1]) if u != v}

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(sorted(edges))

    L_un       = nx.laplacian_matrix(G, nodelist=list(range(n))).astype(float)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.array(L_un.diagonal()) + 1))
    L_norm     = D_inv_sqrt @ L_un @ D_inv_sqrt
    diag_vals  = np.array(L_norm.diagonal())
    return 2 * diag_vals - np.array(np.abs(L_norm).sum(axis=1)).ravel()


def _forman_eigs_filename(args, layer, fold):
    return f"{args['model']}_{args['dataset']}_layer{layer}_fold{fold}{_map_tag(args)}_seed{args['seed']}.npy"


def _save_forman_eigs(laplacian_dict, args, fold, eigs_dir, n, d):
    """Compute and save Forman eigenvalues (n,d) float32 .npy from COO Laplacian dict."""
    os.makedirs(eigs_dir, exist_ok=True)
    for layer, lap in laplacian_dict.items():
        lap_idx = lap[0].detach().cpu()
        lap_val = lap[1].detach().cpu()
        eigs    = np.linalg.eigvalsh(_compute_F_diags(lap_idx, lap_val, n, d)).astype(np.float32)
        path    = os.path.join(eigs_dir, _forman_eigs_filename(args, layer, fold))
        np.save(path, eigs)
        print(f"Saved Forman eigs to {path} shape={eigs.shape}")


def _save_forman_eigs_arrays(eigs_dict, args, fold, eigs_dir):
    """Save pre-computed Forman eigenvalue arrays {layer: (n,d)} to .npy files."""
    os.makedirs(eigs_dir, exist_ok=True)
    for layer, eigs in eigs_dict.items():
        path = os.path.join(eigs_dir, _forman_eigs_filename(args, layer, fold))
        np.save(path, eigs)
        print(f"Saved Forman eigs to {path} shape={eigs.shape}")


def _save_f0_top(f0_top, args, fold, base_dir):
    """Save topological Forman curvature array as .npy."""
    os.makedirs(base_dir, exist_ok=True)
    n        = len(f0_top)
    filename = f"f0_top_{args['dataset']}_n{n}_fold{fold}_seed{args['seed']}.npy"
    np.save(os.path.join(base_dir, filename), f0_top)
    print(f"Saved f0_top to {base_dir}/{filename}")


def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])
    n = data.x.size(0)
    d = args['d']

    if args['sheaf_init'] and model_cls in (DiscreteJointSheafDiffusionParams, DiscreteJointSheafDiffusionParamsAlt):
        sheaf_init = precompute_sheaf_mappings(data, args['d'], args)
        model = model_cls(data.edge_index, args, sheaf_init=sheaf_init)
    else:
        model = model_cls(data.edge_index, args)
    model = model.to(args['device'])
    model.save_other_laplacian = args.get('save_others', False)

    sheaf_learner_params, other_params = model.grouped_parameters()
    maps_lr = args['maps_lr'] if args['maps_lr'] is not None else args['lr']
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'lr': maps_lr, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    test_auc = 0.0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    checkpoint_set = set(args.get('checkpoint_epochs') or [])
    best_norms_snapshot    = None
    best_conf_snapshot     = None
    best_saliency_snapshot = None
    best_correct_snapshot  = None
    best_state_dict        = None
    checkpoint_accuracy = {}   # {label: {train_acc, val_acc, test_acc}}
    best_acc_snapshot = {}

    save_laps     = args.get('save_laplacians', True)
    save_norms    = args.get('save_norms', True)
    save_preds    = args.get('save_preds', True)
    save_saliency = args.get('save_saliency', True)
    save_others   = args.get('save_others', False)

    # Build topological reference once per fold (pure graph topology, no training).
    f0_top = _build_f0_top(data.edge_index, n) if save_laps else None

    # Epoch 0: save pre-training Laplacian/norms (random init, zero gradient steps).
    # test() populates _last_laplacian/_last_node_norms and gives accuracy at no extra cost.
    if 0 in checkpoint_set:
        [ta0, va0, tta0], _, _, _ = test(model, data)
        if save_laps and hasattr(model, '_last_laplacian') \
                and model._last_laplacian[0] is not None \
                and model._last_laplacian[1] is not None:
            _save_forman_eigs(
                model._last_laplacian, args, fold,
                os.path.join(_forman_eigs_base_dir(args), 'checkpoints', 'epoch-0'),
                n, d,
            )
            if save_others and hasattr(model, '_last_laplacian_other') and model._last_laplacian_other:
                _save_forman_eigs(
                    model._last_laplacian_other, args, fold,
                    os.path.join(_forman_eigs_other_base_dir(args), 'checkpoints', 'epoch-0'),
                    n, d,
                )
        if save_norms and hasattr(model, '_last_node_norms') and model._last_node_norms:
            _save_norms(
                model._last_node_norms, args, fold,
                os.path.join(_norms_base_dir(args), 'checkpoints', 'epoch-0'),
            )
        if save_preds or save_saliency:
            _pred0, _conf0, _sal0 = _compute_conf_saliency(model, data)
            if save_preds:
                _save_conf(_conf0, args, fold,
                           os.path.join(_conf_base_dir(args), 'checkpoints', 'epoch-0'))
                _save_acc(_pred0.eq(data.y.cpu()), args, fold,
                          os.path.join(_acc_base_dir(args), 'checkpoints', 'epoch-0'))
            if save_saliency:
                _save_saliency(_sal0, args, fold,
                               os.path.join(_saliency_base_dir(args), 'checkpoints', 'epoch-0'))
        checkpoint_accuracy['epoch-0'] = {
            'train_acc': float(ta0), 'val_acc': float(va0), 'test_acc': float(tta0),
        }

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss], probs = test(model, data)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            wandb.log(res_dict, step=epoch)

        # Checkpoint save: epoch N = after N gradient steps (loop variable = N-1 here).
        if checkpoint_set and (epoch + 1) in checkpoint_set:
            if save_laps and hasattr(model, '_last_laplacian') \
                    and model._last_laplacian[0] is not None \
                    and model._last_laplacian[1] is not None:
                _save_forman_eigs(
                    model._last_laplacian, args, fold,
                    os.path.join(_forman_eigs_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'),
                    n, d,
                )
                if save_others and hasattr(model, '_last_laplacian_other') and model._last_laplacian_other:
                    _save_forman_eigs(
                        model._last_laplacian_other, args, fold,
                        os.path.join(_forman_eigs_other_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'),
                        n, d,
                    )
            if save_norms and hasattr(model, '_last_node_norms') and model._last_node_norms:
                _save_norms(
                    model._last_node_norms, args, fold,
                    os.path.join(_norms_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'),
                )
            if save_preds or save_saliency:
                _pred_ck, _conf_ck, _sal_ck = _compute_conf_saliency(model, data)
                if save_preds:
                    _save_conf(_conf_ck, args, fold,
                               os.path.join(_conf_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'))
                    _save_acc(_pred_ck.eq(data.y.cpu()), args, fold,
                              os.path.join(_acc_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'))
                if save_saliency:
                    _save_saliency(_sal_ck, args, fold,
                                   os.path.join(_saliency_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'))
            checkpoint_accuracy[f'epoch-{epoch + 1}'] = {
                'train_acc': float(train_acc), 'val_acc': float(val_acc), 'test_acc': float(tmp_test_acc),
            }

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_norms and hasattr(model, '_last_node_norms') and model._last_node_norms:
                best_norms_snapshot = {
                    layer: norms.detach().clone().cpu()
                    for layer, norms in model._last_node_norms.items()
                }
            if save_preds or save_saliency:
                _best_pred, _best_conf, _best_sal = _compute_conf_saliency(model, data)
                if save_preds:
                    best_conf_snapshot    = _best_conf
                    best_correct_snapshot = _best_pred.eq(data.y.cpu())
                if save_saliency:
                    best_saliency_snapshot = _best_sal
            best_acc_snapshot = {
                'train_acc': float(train_acc), 'val_acc': float(val_acc), 'test_acc': float(tmp_test_acc),
            }
            if args['dataset'] in AUC_DATASETS:
                y_true = data.y[data.test_mask].cpu().numpy()
                y_probs = probs[2].cpu().numpy()
                try:
                    test_auc = roc_auc_score(y_true, y_probs[:,1])
                except ValueError:
                    test_auc = float('nan')
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    last_test_acc = tmp_test_acc
    if args['dataset'] in AUC_DATASETS:
        y_true_last = data.y[data.test_mask].cpu().numpy()
        y_probs_last = probs[2].cpu().numpy()
        try:
            last_test_auc = roc_auc_score(y_true_last, y_probs_last[:, 1])
        except ValueError:
            last_test_auc = float('nan')

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc (best epoch): {test_acc:.4f}")
    print(f"Test acc (last epoch): {last_test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    if args['dataset'] in AUC_DATASETS:
        print(f"Test AUC (best epoch): {test_auc:.4f}")
        print(f"Test AUC (last epoch): {last_test_auc:.4f}")

    if "ODE" not in args['model']:
        # Debugging for discrete models
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args['layers']):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")
            if hasattr(model, 'dual_epsilons'):
                for i in range(0, args['layers'] - 1):
                    print(f"DualEpsilons {i}: {model.dual_epsilons[i].detach().cpu().numpy().flatten()}")

    if best_norms_snapshot is not None:
        _save_norms(
            best_norms_snapshot, args, fold,
            os.path.join(_norms_base_dir(args), 'checkpoints', 'best-epoch'),
        )
        print(f"Saved best-epoch node norms (epoch {best_epoch}) to checkpoints/best-epoch/")
    if best_conf_snapshot is not None:
        _save_conf(best_conf_snapshot, args, fold,
                   os.path.join(_conf_base_dir(args), 'checkpoints', 'best-epoch'))
    if best_correct_snapshot is not None:
        _save_acc(best_correct_snapshot, args, fold,
                  os.path.join(_acc_base_dir(args), 'checkpoints', 'best-epoch'))
    if best_saliency_snapshot is not None:
        _save_saliency(best_saliency_snapshot, args, fold,
                       os.path.join(_saliency_base_dir(args), 'checkpoints', 'best-epoch'))
    _save_training_args(args, _weights_base_dir(args))
    if best_state_dict is not None:
        _ckpt_dir = os.path.join(_weights_base_dir(args), 'checkpoints', 'best-epoch')
        _save_weights(best_state_dict, args, fold, _ckpt_dir)
        _save_masks(data, args, fold, _ckpt_dir)
        print(f"Saved best-epoch model weights and masks (epoch {best_epoch}) to checkpoints/best-epoch/")

    # Record last-epoch accuracy now that the loop has finished.
    checkpoint_accuracy['last'] = {
        'epoch': int(epoch),
        'train_acc': float(train_acc), 'val_acc': float(val_acc), 'test_acc': float(last_test_acc),
    }
    if best_acc_snapshot:
        checkpoint_accuracy['best-epoch'] = dict(best_acc_snapshot, epoch=int(best_epoch))

    # Accumulate per-fold accuracy and best-epoch info into a single JSON file.
    best_epochs_path = os.path.join(_forman_eigs_base_dir(args), 'best_epochs.json')
    os.makedirs(_forman_eigs_base_dir(args), exist_ok=True)
    best_epochs_data = {}
    if os.path.exists(best_epochs_path):
        with open(best_epochs_path, 'r') as f:
            best_epochs_data = json.load(f)
    best_epochs_data[f'fold_{fold}'] = {
        'best_epoch': int(best_epoch),
        'best_val_acc': float(best_val_acc),
        'best_test_acc': float(test_acc),
        'checkpoints': checkpoint_accuracy,
    }
    with open(best_epochs_path, 'w') as f:
        json.dump(best_epochs_data, f, indent=2)
    print(f"Updated best_epochs.json: fold_{fold} -> epoch {best_epoch}")

    # Save Forman eigenvalues and f0_top from the last epoch.
    if save_laps and hasattr(model, '_last_laplacian') \
        and model._last_laplacian[0] is not None \
        and model._last_laplacian[1] is not None:
        _save_forman_eigs(model._last_laplacian, args, fold, _forman_eigs_base_dir(args), n, d)
        if save_others and hasattr(model, '_last_laplacian_other') and model._last_laplacian_other:
            _save_forman_eigs(model._last_laplacian_other, args, fold, _forman_eigs_other_base_dir(args), n, d)
        _save_f0_top(f0_top, args, fold, _forman_eigs_base_dir(args))

    if save_norms and hasattr(model, '_last_node_norms') and model._last_node_norms:
        _save_norms(model._last_node_norms, args, fold, _norms_base_dir(args))

    if save_preds or save_saliency:
        _pred_last, _conf_last, _sal_last = _compute_conf_saliency(model, data)
        if save_preds:
            _save_node_preds(
                {
                    'pred':       _pred_last,
                    'conf':       _conf_last,
                    'correct':    _pred_last.eq(data.y.cpu()),
                    'y':          data.y.cpu(),
                    'train_mask': data.train_mask.cpu(),
                    'val_mask':   data.val_mask.cpu(),
                    'test_mask':  data.test_mask.cpu(),
                },
                args, fold, _preds_base_dir(args),
            )
            _save_conf(_conf_last, args, fold, _conf_base_dir(args))
            _save_acc(_pred_last.eq(data.y.cpu()), args, fold, _acc_base_dir(args))
        if save_saliency:
            _save_saliency(_sal_last, args, fold, _saliency_base_dir(args))

    if args['dataset'] in AUC_DATASETS:
        wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch, 'best_test_auc': test_auc,
                   'last_test_acc': last_test_acc, 'last_test_auc': last_test_auc})
    else:
        wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,
                   'last_test_acc': last_test_acc})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, test_auc, keep_running

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args.model == 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args.model == 'JointSheafParams':
        model_cls = DiscreteJointSheafDiffusionParams
    elif args.model == 'JointSheafParamsAlt':
        model_cls = DiscreteJointSheafDiffusionParamsAlt
    elif args.model == 'JointSheafVanilla':
        model_cls = DiscreteJointSheafVanillaDiffusion
    elif args.model == 'VanillaSheaf':
        model_cls = DiscreteVanillaDiffusion
    elif args.model == 'ConvSheaf':
        model_cls = DiscreteVanillaDiffusionAlt
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = get_dataset(args.dataset,args)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)

    # ADAPTING FROM FERRAN'S CODE
    #args.input_dim = dataset.num_features
    #args.output_dim = dataset.num_classes
    args.input_dim = dataset[0].x.shape[1]          # ← already fixed (same as my suggestion)
    try:
        args.output_dim = dataset.num_classes        # ← tries the InMemoryDataset property first
    except: 
        args.output_dim = torch.unique(dataset[0].y).shape[0]  # ← fallback for plain lists


    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # I AM COMMENTING THIS TO TRY UNNORMALISATION
    # assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    map_suffix = _map_tag(vars(args))
    if args.dataset == "synthetic_exp":
        run_name = f"{args.model}_nodes-{args.num_nodes}_node-deg-{args.node_degree}_normalised-{str(args.normalised).lower()}_stalk-{args.d}_{args.layers}layers_{args.hidden_channels}hidden_{args.epochs}epochs_pct-hetero-{int(float(args.het_coef)*100)}_classes-{args.num_classes}_feats-{args.num_feats}{map_suffix}_seed{args.seed}"
    else:
        run_name = f"{args.model}_{args.dataset}_normalised-{str(args.normalised).lower()}_stalk-{args.d}_{args.layers}layers_{args.hidden_channels}hidden_{args.epochs}epochs{map_suffix}_seed{args.seed}"
    wandb.init(project="sheaf", config=vars(args), entity=args.entity, name=run_name)

    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, test_auc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc, test_auc])
        if not keep_running:
            break

    # if hasattr(model_cls, '_last_maps') and model_cls._last_maps is not None:
    #     maps_dir = r"results/maps"
    #     os.makedirs(maps_dir, exist_ok=True)

    #     maps_filename = f"{args['model']}_{args['dataset']}_fold{fold}_seed{args['seed']}.pt"
    #     maps_path = os.path.abspath(os.path.join(maps_dir, maps_filename))
    #     torch.save(model_cls._last_maps.detach().cpu(), maps_path)
    #     print(f"Saved last restriction maps to {maps_path}")

    results_arr = np.array(results)
    test_acc_mean = results_arr[:, 0].mean() * 100
    val_acc_mean  = results_arr[:, 1].mean() * 100
    test_acc_std  = results_arr[:, 0].std() * 100

    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}

    if args.dataset in AUC_DATASETS:
        test_auc_mean = results_arr[:, 2].mean()
        test_auc_std  = results_arr[:, 2].std()
        wandb_results['test_auc']     = test_auc_mean
        wandb_results['test_auc_std'] = test_auc_std

    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
    if args.dataset in AUC_DATASETS:
        print(f'Test AUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}')

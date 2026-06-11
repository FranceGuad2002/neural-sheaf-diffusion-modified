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


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds, probs = model(data.x), [], [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
            probs.append(torch.softmax(logits[mask], dim = 1).detach().cpu())
        return accs, preds, losses, probs


def _lap_base_dir(args):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'laplacians',
        args['dataset'],
        f"normalised-{str(args['normalised']).lower()}",
        f"stalk_dim-{args['d']}",
        f"{args['layers']}-layers",
        f"{args['hidden_channels']}-hidden",
        f"{args['epochs']}-epochs",
    ))


def _save_laplacians(laplacian_dict, args, fold, lap_dir):
    os.makedirs(lap_dir, exist_ok=True)
    lfm_tag = (f"_learn_first_maps-{str(args['learn_first_maps']).lower()}"
               if args['model'] == 'JointSheafParamsAlt' else "")

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
                f"{lfm_tag}_seed{args['seed']}.pt"
            )
        else:
            lap_filename = (
                f"{args['model']}_{args['dataset']}_layer{layer}"
                f"_fold{fold}{lfm_tag}_seed{args['seed']}.pt"
            )

        lap_path = os.path.join(lap_dir, lap_filename)
        torch.save(lap_matrix, lap_path)
        print(f"Saved Laplacian to {lap_path} with shape {tuple(lap_matrix.shape)}")


def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    if args['sheaf_init'] and model_cls in (DiscreteJointSheafDiffusionParams, DiscreteJointSheafDiffusionParamsAlt):
        sheaf_init = precompute_sheaf_mappings(data, args['d'], args)
        model = model_cls(data.edge_index, args, sheaf_init=sheaf_init)
    else:
        model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

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
    best_laplacian_snapshot = None
    checkpoint_accuracy = {}   # {label: {train_acc, val_acc, test_acc}}
    best_acc_snapshot = {}

    # Epoch 0: save pre-training Laplacian (random init, zero gradient steps).
    # test() populates _last_laplacian and gives accuracy at no extra cost.
    if 0 in checkpoint_set:
        [ta0, va0, tta0], _, _, _ = test(model, data)
        if hasattr(model, '_last_laplacian') \
                and model._last_laplacian[0] is not None \
                and model._last_laplacian[1] is not None:
            _save_laplacians(
                model._last_laplacian, args, fold,
                os.path.join(_lap_base_dir(args), 'checkpoints', 'epoch-0'),
            )
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
            if hasattr(model, '_last_laplacian') \
                    and model._last_laplacian[0] is not None \
                    and model._last_laplacian[1] is not None:
                _save_laplacians(
                    model._last_laplacian, args, fold,
                    os.path.join(_lap_base_dir(args), 'checkpoints', f'epoch-{epoch + 1}'),
                )
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
            # Snapshot the Laplacian at the best validation epoch for later saving.
            if hasattr(model, '_last_laplacian') \
                    and model._last_laplacian[0] is not None \
                    and model._last_laplacian[1] is not None:
                best_laplacian_snapshot = {
                    layer: (lap[0].detach().clone().cpu(), lap[1].detach().clone().cpu())
                    for layer, lap in model._last_laplacian.items()
                }
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

    # Save the Laplacian from the best validation epoch (snapshotted during training).
    if best_laplacian_snapshot is not None:
        _save_laplacians(
            best_laplacian_snapshot, args, fold,
            os.path.join(_lap_base_dir(args), 'checkpoints', 'best-epoch'),
        )
        print(f"Saved best-epoch Laplacian (epoch {best_epoch}) to checkpoints/best-epoch/")

    # Record last-epoch accuracy now that the loop has finished.
    checkpoint_accuracy['last'] = {
        'epoch': int(epoch),
        'train_acc': float(train_acc), 'val_acc': float(val_acc), 'test_acc': float(last_test_acc),
    }
    if best_acc_snapshot:
        checkpoint_accuracy['best-epoch'] = dict(best_acc_snapshot, epoch=int(best_epoch))

    # Accumulate per-fold accuracy and best-epoch info into a single JSON file.
    best_epochs_path = os.path.join(_lap_base_dir(args), 'best_epochs.json')
    os.makedirs(_lap_base_dir(args), exist_ok=True)
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

    # Save the Laplacian from the last epoch (existing behaviour, path unchanged).
    if hasattr(model, '_last_laplacian') \
        and model._last_laplacian[0] is not None \
        and model._last_laplacian[1] is not None:
        _save_laplacians(model._last_laplacian, args, fold, _lap_base_dir(args))
    
    if hasattr(model, '_last_node_norms') and model._last_node_norms:
        norms_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'node_norms',
        f'{args["dataset"]}',
        f"normalised-{str(args['normalised']).lower()}",
        f'stalk_dim-{args["d"]}',
        f'{args["layers"]}-layers',
        f'{args["hidden_channels"]}-hidden',
        f'{args["epochs"]}-epochs'))
        os.makedirs(norms_dir, exist_ok=True)

        _lfm_tag = (f"_learn_first_maps-{str(args['learn_first_maps']).lower()}"
                    if args['model'] == 'JointSheafParamsAlt' else "")

        if args["dataset"] == "synthetic_exp":
            norms_filename = f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}_pct-hetero-{int(float(args['het_coef'])*100)}_classes-{args['num_classes']}_feats-{args['num_feats']}{_lfm_tag}_fold{fold}_seed{args['seed']}.pt"
        else:
            norms_filename = f"{args['model']}_{args['dataset']}{_lfm_tag}_fold{fold}_seed{args['seed']}.pt"

        norms_path = os.path.join(norms_dir, norms_filename)
        torch.save(model._last_node_norms, norms_path)
        print(f"Saved node norms to {norms_path}")

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
    lfm_suffix = f"_learn_first_maps-{str(args.learn_first_maps).lower()}" if args.model == 'JointSheafParamsAlt' else ""
    if args.dataset == "synthetic_exp":
        run_name = f"{args.model}_nodes-{args.num_nodes}_node-deg-{args.node_degree}_normalised-{str(args.normalised).lower()}_stalk-{args.d}_{args.layers}layers_{args.hidden_channels}hidden_{args.epochs}epochs_pct-hetero-{int(float(args.het_coef)*100)}_classes-{args.num_classes}_feats-{args.num_feats}{lfm_suffix}_seed{args.seed}"
    else:
        run_name = f"{args.model}_{args.dataset}_normalised-{str(args.normalised).lower()}_stalk-{args.d}_{args.layers}layers_{args.hidden_channels}hidden_{args.epochs}epochs{lfm_suffix}_seed{args.seed}"
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

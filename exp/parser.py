# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from email import parser

from distutils.util import strtobool
import argparse
import os


def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def get_parser():
    parser = argparse.ArgumentParser()
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--sheaf_decay', type=float, default=None)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='loss')

    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False,
                        help="Use a a degree-normalised Laplacian")
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)

    # Experiment parameters
    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', type=str, choices=['DiagSheaf', 'BundleSheaf', 'GeneralSheaf', 'DiagSheafODE',
                                                      'BundleSheafODE', 'GeneralSheafODE', 'JointSheafParams',
                                                      'JointSheafParamsAlt', 'JointSheafVanilla',
                                                      'VanillaSheaf', 'ConvSheaf'], default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--evectors', type=int, default=0, help="Number of Laplacian PE eigenvectors to use.")

    # ODE args
    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")
    parser.add_argument('--int_method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")
    
    ## PARSER FOR SYNTHETIC DATASETS 
    # Synthetic dataset parameters
    parser.add_argument("--num_nodes", type=int, default=200,
                    help="Number of nodes in the synthetic graph.")
    parser.add_argument("--num_classes", type=int, default=2,
                    help="Number of classes in the synthetic graph.")
    parser.add_argument("--num_feats", type=int, default=10,
                    help="Number of node features.")
    parser.add_argument("--het_coef", type=float, default=0.9,
                    help="Heterophily coefficient. 0=homophilic, 1=fully heterophilic.")
    parser.add_argument("--edge_noise", type=float, default=0.05,
                    help="Probability of randomly removing an edge.")
    parser.add_argument("--node_degree", type=int, default=10,
                    help="Average node degree in the synthetic graph.")
    parser.add_argument("--feat_noise", type=float, default=0.25,
                    help="Standard deviation of Gaussian noise added to features.")
    parser.add_argument("--ellipsoid_radius", type=float, default=1,
                    help="Radius of the ellipsoid for feature generation.")
    parser.add_argument("--just_add_noise", type=str2bool, default=False,
                    help="If True, reuse saved features and only add new noise.")
    parser.add_argument("--ellipsoids", type=str2bool, default=True,
                    help="If True, use ellipsoid mode. If False, use Gaussian mode.")
    parser.add_argument("--classes_corr", type=list_of_floats, default=None,
                    help="Custom class correlation matrix, as a flat comma-separated list.")


    # Joint sheaf diffusion args
    parser.add_argument('--dual_diff_strength', type=float, default=1.0,
                        help="Regulates the diffusion strength of the dual sheaf diffusion")
    parser.add_argument('--dual_normalised', type=str2bool, default=True)
    parser.add_argument('--sheaf_init', type=str2bool, default=False,
                        help="Precompute initial sheaf via DiscreteJointSheafVanillaDiffusion + PCA")
    parser.add_argument('--dual_left_linear', type=str2bool, default=False)
    parser.add_argument('--dual_right_linear', type=str2bool, default=False)
    parser.add_argument('--dual_linear', type=str2bool, default=True)
    parser.add_argument('--learn_first_maps', type=str2bool, default=False)
    parser.add_argument('--learnt_map_type', type=str,
                        choices=['diagonal', 'bundle', 'general'], default='diagonal',
                        help="Type of restriction maps to learn as initial maps: "
                             "diagonal, bundle (orthogonal), or general (unconstrained d x d).")
    parser.add_argument('--rotation_invariant_sheaf_learner', type=str2bool, default=False)
    parser.add_argument('--node_edge_sims_time_dependent', type=str2bool, default=False)
    parser.add_argument('--num_of_convolutions', type=int, default=1)
    parser.add_argument('--dim_list', type=list_of_ints, default=[])
    parser.add_argument('--zero_laplacian', type=str2bool, default=False)
    parser.add_argument('--maps_lr', type=float, default=None)
    parser.add_argument('--use_epsilons', type=str2bool, default=True)
    parser.add_argument('--use_embedding', type=str2bool, default=True)

    parser.add_argument('--checkpoint_epochs', type=list_of_ints, default=[],
                        help='Comma-separated epoch numbers at which to save Laplacian checkpoints. '
                             '0 = pre-training (before any gradient step). '
                             'N = after N gradient steps. '
                             'Example: --checkpoint_epochs 0,1,5,15,200')

    parser.add_argument('--save_laplacians', type=str2bool, default=True,
                        help="Save the sheaf Laplacian at checkpoints and at the end of training.")
    parser.add_argument('--save_norms', type=str2bool, default=True,
                        help="Save node hidden-representation norms at checkpoints and at the end of training.")
    parser.add_argument('--save_preds', type=str2bool, default=True,
                        help="Save per-node predictions, confidence, and correctness at the end of training.")
    parser.add_argument('--save_saliency', type=str2bool, default=True,
                        help="Save per-node input-gradient saliency (||dL/dX_i||) at the end of training.")
    parser.add_argument('--save_others', type=str2bool, default=False,
                        help="(GeneralSheaf only) Also save Forman eigs of the opposite-normalisation Laplacian "
                             "to other_Laplacian/ alongside the primary eigs.")

    return parser
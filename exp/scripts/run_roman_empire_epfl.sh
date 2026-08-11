#!/bin/bash
#SBATCH --job-name=sheaf_diffusion
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export WANDB_MODE=online
export WANDB_ENTITY=franceguad2002-epfl
export WANDB_PROJECT=sheaf
# API key intentionally not stored here: run `wandb login` once; it writes
# ~/.netrc (mode 0600), which SLURM jobs inherit. Never hardcode the key.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nsd

ARGS=(
    # --- experiment [all models] ---
    --dataset=roman_empire
    --folds=10
    --epochs=500
    --early_stopping=200

    # --- model [all models] ---
    --model=GeneralSheaf

    # --- sheaf architecture [all models] ---
    --d=5
    --hidden_channels=16
    --layers=5
    --add_hp=False
    --add_lp=False
    --normalised=True
    --deg_normalised=False
    --left_weights=True
    --right_weights=True
    --use_act=True

    # --- sheaf learner [DiagSheaf, BundleSheaf, GeneralSheaf] ---
    --sparse_learner=False

    # --- connection Laplacian [BundleSheaf, JointSheafParams*] ---
    --edge_weights=True

    # --- optimizer [all models] ---
    --lr=0.01
    --weight_decay=0.0006914841722570725
    --sheaf_decay=0.00031764232712732976

    # --- regularisation [all models] ---
    --dropout=0.3
    --input_dropout=0.2

    # --- residual epsilons [VanillaSheaf, ConvSheaf, JointSheafParams*] ---
    # (DiagSheaf / BundleSheaf / GeneralSheaf always train epsilons — flag has no effect there)
    --use_epsilons=True

    # --- joint diffusion only [JointSheafParams, JointSheafParamsAlt] ---
    --dual_normalised=True
    --dual_diff_strength=1.0
    --dual_linear=False
    --dual_left_linear=True
    --dual_right_linear=True
    --learn_first_maps=False
    --learnt_map_type=general
    --sheaf_init=False

    # --- VanillaSheaf only ---
    --use_embedding=True

    # --- logging [all models] ---
    --checkpoint_epochs=0,1,5,15,50,200
    --save_laplacians=False
    # save_others: disabled for JointSheafParams (identity maps, no other_Laplacian needed)
    --save_others=False
    --save_norms=False
    --save_preds=False
    --save_saliency=False
    --entity="${WANDB_ENTITY}"
)

python -m exp.run "${ARGS[@]}"

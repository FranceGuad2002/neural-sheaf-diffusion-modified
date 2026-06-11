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
export WANDB_API_KEY=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


# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nsd

# Run your experiment
python -m exp.run \
    --add_hp=False \
    --add_lp=False \
    --d=5 \
    --dataset=roman_empire \
    --dropout=0.3 \
    --early_stopping=200 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=16 \
    --input_dropout=0.2 \
    --layers=5 \
    --lr=0.01 \
    --model=GeneralSheaf \
    --sheaf_decay=0.00031764232712732976 \
    --weight_decay=0.0006914841722570725 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=False \
    --sparse_learner=False \
    --deg_normalised=False \
    --dual_normalised=True \
    --dual_diff_strength=1.0 \
    --use_epsilons=True \
    --dual_linear=True \
    --dual_left_linear=False \
    --dual_right_linear=False \
    --learn_first_maps=False \
    --dual_diag=False \
    --sheaf_init=False \
    --use_embedding=True \
    --checkpoint_epochs=0,1,5,15,200 \
    --entity="${WANDB_ENTITY}"
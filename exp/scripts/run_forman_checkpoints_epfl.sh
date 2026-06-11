#!/bin/bash
#SBATCH --job-name=forman_checkpoints
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Do_nsd

# Run Forman Profile checkpoint evolution for all combinations with a checkpoints/ dir
python quick_analysis/compute_forman_checkpoints.py

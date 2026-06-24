#!/bin/bash
#SBATCH --job-name=forman_metrics
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── analysis config ────────────────────────────────────────────────────────────
NORMALISED="true"          # "true" or "false"
MODEL="GeneralSheaf"       # e.g. GeneralSheaf | JointSheafParamsAlt
LEARN_FIRST_MAPS="false"   # only matters when MODEL=JointSheafParamsAlt

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Do_nsd

python quick_analysis/compute_forman_metrics.py \
    --normalised="${NORMALISED}" \
    --model="${MODEL}" \
    --learn_first_maps="${LEARN_FIRST_MAPS}"

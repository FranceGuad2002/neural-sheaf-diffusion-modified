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

# ── analysis config ────────────────────────────────────────────────────────────
NORMALISED="true"           # "true" or "false" — which normalised-* folder to scan
MODEL="GeneralSheaf"        # e.g. GeneralSheaf | JointSheafParams | JointSheafParamsAlt
MAP_TYPE="identity"         # "identity" | "general" | "diag" | "bundle" — Joint models only

# Set to "true" to read eigs from other_Laplacian/ subfolders (opposite-normalisation content).
# The output folder will gain a _other suffix so it never collides with the primary run.
USE_OTHER="false"

# Set to "true" to run the cross-comparison pipeline (both directions).
# Ignores NORMALISED and USE_OTHER — always loads both norm=true and norm=false data.
DO_OTHER_COMPARISON="false"

# ── build optional flags ───────────────────────────────────────────────────────
USE_OTHER_FLAG=""
[ "${USE_OTHER}" = "true" ] && USE_OTHER_FLAG="--use_other"

DO_COMPARISON_FLAG=""
[ "${DO_OTHER_COMPARISON}" = "true" ] && DO_COMPARISON_FLAG="--do_other_comparison"

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Do_nsd

python quick_analysis/compute_forman_checkpoints.py \
    --normalised="${NORMALISED}" \
    --model="${MODEL}" \
    --map_type="${MAP_TYPE}" \
    ${USE_OTHER_FLAG} \
    ${DO_COMPARISON_FLAG}

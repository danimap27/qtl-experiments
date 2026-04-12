#!/bin/bash
#SBATCH --job-name=qtl_noise_decomp
#SBATCH --output=logs/noise_%A_%a.out
#SBATCH --error=logs/noise_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-48%20

# ============================================================
# QTL Paper v2 — Descomposición de ruido por canal (48 runs)
# Lanzar con: sbatch slurm_noise.sh
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qtl

cd $SLURM_SUBMIT_DIR

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm_commands_noise.txt)
echo "[$SLURM_ARRAY_TASK_ID] Running: $CMD"
eval "$CMD --machine-id hercules_noise"

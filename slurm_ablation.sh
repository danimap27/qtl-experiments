#!/bin/bash
#SBATCH --job-name=qtl_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-32%20

# ============================================================
# QTL Paper v2 — Ablación qubits/profundidad (32 runs)
# Lanzar con: sbatch slurm_ablation.sh
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qtl

cd $SLURM_SUBMIT_DIR

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm_commands_ablation.txt)
echo "[$SLURM_ARRAY_TASK_ID] Running: $CMD"
eval "$CMD --machine-id hercules_ablation"

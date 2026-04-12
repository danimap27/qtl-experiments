#!/bin/bash
#SBATCH --job-name=qtl_paper_v2
#SBATCH --output=logs/qtl_%A_%a.out
#SBATCH --error=logs/qtl_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-560%40

# ============================================================
# QTL Paper v2 — Batería principal (560 runs, 5 seeds)
# Lanzar con: sbatch slurm_main.sh
# ============================================================

# Activar entorno
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qtl

cd $SLURM_SUBMIT_DIR

# Generar comandos y ejecutar el de este array task
COMMANDS_FILE="slurm_commands_main.txt"
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $COMMANDS_FILE)

echo "[$SLURM_ARRAY_TASK_ID] Running: $CMD"
eval "$CMD --machine-id hercules_v2"

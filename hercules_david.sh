#!/bin/bash
# hercules_david.sh — SLURM job array for David's account
# Runs: QK-Noisy (80) + sim=HW noisy QK (4) = 84 runs
#
# Before launching:
#   source $HOME/qtl_venv/bin/activate
#   cd $HOME/qtl_experiments
#   python runner.py --config config.yaml --machine-id hercules_david \
#       --head qk_noisy --dry-run --export-commands > hercules_david_runs.txt
#   python runner.py --config config.yaml --machine-id hercules_david \
#       --study sim_as_hardware --head qk_noisy --dry-run --export-commands >> hercules_david_runs.txt
#   wc -l hercules_david_runs.txt   # expected: 84

#SBATCH --job-name=qtl_david
#SBATCH --output=results/slurm_david_%A_%a.out
#SBATCH --error=results/slurm_david_%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-84

# =============================================
# Environment
# =============================================
WORK_DIR="$HOME/qtl_experiments"
VENV_DIR="$HOME/qtl_venv"
RUN_LIST="$WORK_DIR/hercules_david_runs.txt"

cd "$WORK_DIR"

module load Anaconda3/2022.05 2>/dev/null \
    || module load Python/3.9.6-GCCcore-11.2.0 2>/dev/null \
    || true

source "$VENV_DIR/bin/activate"

# =============================================
# Execute the run for this array index
# =============================================
RUN_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN_LIST")

echo "[HERCULES_DAVID] Array task $SLURM_ARRAY_TASK_ID of ${SLURM_ARRAY_TASK_COUNT:-84}"
echo "[HERCULES_DAVID] Running: $RUN_CMD"
echo "[HERCULES_DAVID] Node: $(hostname)"
echo "[HERCULES_DAVID] Start: $(date)"

eval "$RUN_CMD"
EXIT_CODE=$?

echo "[HERCULES_DAVID] Exit code: $EXIT_CODE"
echo "[HERCULES_DAVID] End: $(date)"

exit $EXIT_CODE

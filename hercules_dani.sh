#!/bin/bash
# hercules_dani.sh — SLURM job array for Dani's account
# Runs: PL-Noisy (80) + noise decomposition (48) + sim=HW noisy PL (4) = 132 runs
#
# Before launching:
#   source $HOME/qtl_venv/bin/activate
#   cd $HOME/qtl_experiments
#   python runner.py --config config.yaml --machine-id hercules_dani \
#       --head pl_noisy --dry-run --export-commands > hercules_dani_runs.txt
#   python runner.py --config config.yaml --machine-id hercules_dani \
#       --study noise_decomposition --dry-run --export-commands >> hercules_dani_runs.txt
#   python runner.py --config config.yaml --machine-id hercules_dani \
#       --study sim_as_hardware --head pl_noisy --dry-run --export-commands >> hercules_dani_runs.txt
#   wc -l hercules_dani_runs.txt   # expected: 132

#SBATCH --job-name=qtl_dani
#SBATCH --output=results/slurm_dani_%A_%a.out
#SBATCH --error=results/slurm_dani_%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-132

# =============================================
# Environment
# =============================================
WORK_DIR="$HOME/qtl_experiments"
VENV_DIR="$HOME/qtl_venv"
RUN_LIST="$WORK_DIR/hercules_dani_runs.txt"

cd "$WORK_DIR"

module load Anaconda3/2022.05 2>/dev/null \
    || module load Python/3.9.6-GCCcore-11.2.0 2>/dev/null \
    || true

source "$VENV_DIR/bin/activate"

# =============================================
# Execute the run for this array index
# =============================================
RUN_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN_LIST")

echo "[HERCULES_DANI] Array task $SLURM_ARRAY_TASK_ID of ${SLURM_ARRAY_TASK_COUNT:-132}"
echo "[HERCULES_DANI] Running: $RUN_CMD"
echo "[HERCULES_DANI] Node: $(hostname)"
echo "[HERCULES_DANI] Start: $(date)"

eval "$RUN_CMD"
EXIT_CODE=$?

echo "[HERCULES_DANI] Exit code: $EXIT_CODE"
echo "[HERCULES_DANI] End: $(date)"

exit $EXIT_CODE

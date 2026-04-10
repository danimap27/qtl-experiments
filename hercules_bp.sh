#!/bin/bash
# hercules_bp.sh — SLURM job array for Barren Plateau analysis
# 6 configs: pl_ideal × {2,4,6} qubits + pl_noisy × {2,4,6} qubits
# Each runs 200 random initializations — NOT training runs.

#SBATCH --job-name=qtl_bp
#SBATCH --output=results/slurm_bp_%A_%a.out
#SBATCH --error=results/slurm_bp_%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-6

# =============================================
# Environment
# =============================================
WORK_DIR="$HOME/qtl_experiments"
VENV_DIR="$HOME/qtl_venv"

cd "$WORK_DIR"

module load Anaconda3/2022.05 2>/dev/null \
    || module load Python/3.9.6-GCCcore-11.2.0 2>/dev/null \
    || true

source "$VENV_DIR/bin/activate"

# =============================================
# Map array index to (head, qubits)
# =============================================
case $SLURM_ARRAY_TASK_ID in
    1) HEAD="pl_ideal"; QUBITS=2 ;;
    2) HEAD="pl_ideal"; QUBITS=4 ;;
    3) HEAD="pl_ideal"; QUBITS=6 ;;
    4) HEAD="pl_noisy"; QUBITS=2 ;;
    5) HEAD="pl_noisy"; QUBITS=4 ;;
    6) HEAD="pl_noisy"; QUBITS=6 ;;
esac

echo "[HERCULES_BP] Config $SLURM_ARRAY_TASK_ID: head=$HEAD, qubits=$QUBITS"
echo "[HERCULES_BP] Node: $(hostname)"
echo "[HERCULES_BP] Start: $(date)"

python bp_analysis.py \
    --config config.yaml \
    --machine-id hercules_dani \
    --head "$HEAD" \
    --qubits "$QUBITS"

EXIT_CODE=$?

echo "[HERCULES_BP] Exit code: $EXIT_CODE"
echo "[HERCULES_BP] End: $(date)"

exit $EXIT_CODE

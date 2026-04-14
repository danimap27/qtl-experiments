#!/bin/bash
# slurm_generic.sh
# Generic SLURM template for QTL job arrays.
# Variables passed via --export: CMD_FILE

#SBATCH --output=logs/qtl_%A_%a.out
#SBATCH --error=logs/qtl_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 1. Environment Activation
# Explicit path for Hercules Cluster setup
CONDA_BASE="/lustre/software/easybuild/common/software/Miniconda3/4.9.2"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate qtl
else
    # Fallback to auto-detection if the above fails
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate qtl
fi

# 2. Workspace Setup
cd $SLURM_SUBMIT_DIR

# 3. Command Selection
# Get the specific command for this array index from the passed CMD_FILE
if [ -z "$CMD_FILE" ]; then
    echo "ERROR: CMD_FILE not specified."
    exit 1
fi

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")

# 4. Execution
echo "--------------------------------------------------------"
echo "Job ID: $SLURM_ARRAY_JOB_ID | Task: $SLURM_ARRAY_TASK_ID"
echo "Command File: $CMD_FILE"
echo "Executing: $CMD"
echo "--------------------------------------------------------"

# Run the command
# We append --machine-id hercules for tracking
eval "$CMD --machine-id hercules_v2"

echo "--------------------------------------------------------"
echo "Execution finished at $(date)"
echo "--------------------------------------------------------"

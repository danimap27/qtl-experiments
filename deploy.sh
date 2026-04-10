#!/bin/bash
# deploy.sh — Deploy the QTL experimentation framework
# Usage:
#   ./deploy.sh              # Standard (lab PCs)
#   ./deploy.sh --hercules   # HERCULES (CICA)
#   ./deploy.sh --verify     # Verify installation
set -e

REPO_URL="https://github.com/Data-Science-Big-Data-Research-Lab/QTL.git"
BRANCH="revision-v2"
VENV_DIR=".venv"

HERCULES=false
VERIFY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --hercules) HERCULES=true; shift;;
        --verify)   VERIFY=true; shift;;
        *)          echo "Unknown argument: $1"; exit 1;;
    esac
done

# =============================================
# --verify: check that everything is installed
# =============================================
if [ "$VERIFY" = true ]; then
    echo "=== Environment Verification ==="
    echo -n "Python: ";      python3 --version
    echo -n "PyTorch: ";     python3 -c "import torch; print(torch.__version__)"
    echo -n "PennyLane: ";   python3 -c "import pennylane; print(pennylane.__version__)"
    echo -n "Qiskit: ";      python3 -c "import qiskit; print(qiskit.__version__)"
    echo -n "Qiskit Aer: ";  python3 -c "import qiskit_aer; print(qiskit_aer.__version__)"
    echo -n "Qiskit ML: ";   python3 -c "import qiskit_machine_learning; print(qiskit_machine_learning.__version__)"
    echo -n "CodeCarbon: ";  python3 -c "import codecarbon; print(codecarbon.__version__)"
    echo -n "scikit-learn: ";python3 -c "import sklearn; print(sklearn.__version__)"
    echo -n "pandas: ";      python3 -c "import pandas; print(pandas.__version__)"
    echo ""
    python3 runner.py --config config.yaml --dry-run --count
    echo "=== Verification complete ==="
    exit 0
fi

# =============================================
# Clone or update repo
# =============================================
if [ -d "qtl_experiments" ]; then
    echo "Updating repo..."
    cd qtl_experiments && git pull origin "$BRANCH"
else
    echo "Cloning repo..."
    git clone -b "$BRANCH" "$REPO_URL" qtl_experiments && cd qtl_experiments
fi

# =============================================
# HERCULES-specific setup
# =============================================
if [ "$HERCULES" = true ]; then
    echo "Configuring for HERCULES (CICA)..."
    module load Anaconda3/2022.05 2>/dev/null \
        || module load Python/3.9.6-GCCcore-11.2.0 2>/dev/null \
        || true
    echo "Python loaded: $(python3 --version)"

    VENV_DIR="$HOME/qtl_venv"
    SCRATCH_DIR="/lustre/scratch/$USER/qtl_results"
    echo "Creating scratch dir: $SCRATCH_DIR"
    mkdir -p "$SCRATCH_DIR"
    ln -sf "$SCRATCH_DIR" results
fi

# =============================================
# Virtual environment + dependencies
# =============================================
echo "Creating virtual environment in $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p results data/datasets

# =============================================
# Dataset check
# =============================================
echo ""
echo "=== Dataset check ==="
for ds in hymenoptera brain_tumor cats_vs_dogs solar_dust; do
    if [ -d "data/datasets/$ds" ]; then
        echo "  $ds: OK"
    else
        echo "  $ds: NOT FOUND — run: python data/download_datasets.py --dataset $ds"
    fi
done

# =============================================
# Summary
# =============================================
echo ""
python3 runner.py --config config.yaml --dry-run --count
echo ""
echo "=== Deployment complete ==="
echo "To activate:  source $VENV_DIR/bin/activate"
echo "To run:       python runner.py --config config.yaml --machine-id <id>"

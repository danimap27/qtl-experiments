# Hercules Cluster Deployment Guide: QTL Benchmarking

This guide provides step-by-step instructions for launching the Quantum Transfer Learning (QTL) experiments on the Hercules Cluster.

## 1. Environment Preparation

Before launching experiments, ensure your Conda environment is correctly configured on the cluster.

```bash
# Create a new environment
conda create -n qtl python=3.10 -y
conda activate qtl

# Install core dependencies
pip install torch torchvision torchaudio
pip install pennylane pennylane-lightning[gpu]
pip install qiskit qiskit-aer-gpu
pip install pandas pyyaml matplotlib codecarbon
```

> [!IMPORTANT]
> For GPU acceleration in PennyLane, ensure `pennylane-lightning[gpu]` is installed with the appropriate cuQuantum libraries available in your cluster modules.

## 2. Deploying the Code

Upload the project folder to your home directory or work partition on Hercules.

```bash
# Example using rsync from your local machine
rsync -avz ./qtl_experiments your_user@hercules-cluster-ip:~/
```

## 3. Launching the Management HUB

We have provided a unified management interface (`manager.py`) to handle all SLURM job submissions and monitoring.

```bash
# Navigate to the project directory
cd ~/qtl_experiments

# Launch the HUB
python manager.py
```

### Management Options:
- **`[R] Refresh`**: Run this first to ensure `cmds_*.txt` are synced with the latest `config.yaml`.
- **`[F] Full Pipeline`**: The recommended option. It submits all 692 experiments to the SLURM queue with automatic dependencies (Classical -> Ideal -> Noisy -> Studies).
- **`[1-4] Phase Launch`**: Use these to run specific parts of the benchmark if you are debugging or only need certain results.

## 4. Monitoring & Tracking

You can monitor the status of your jobs directly from the HUB or via standard SLURM commands.

- **Option `[M]` in the HUB**: Scans the `results/` directory and shows a percentage-based progress report by counting completed `runs.csv` files.
- **CLI Commands**:
  ```bash
  squeue -u $USER          # Check active jobs in the queue
  tail -f logs/qtl_*.err   # Monitor real-time logs for a specific task
  ```

## 5. Result Extraction & Table Generation

Once the "Overall Progress" hits 100% in the monitor:

1. Launch `python manager.py`.
2. Select option **`[T] Generate LaTeX Results Tables`**.
3. Your manuscript-ready tables will be located in: `paper/tables/`.

---
*Developed for the CMES Journal Submission Revision (April 2026).*

# Hybrid Classical-Quantum Transfer Learning Benchmarking Framework

## Overview
This platform is a comprehensive benchmarking suite designed for the evaluation of **Hybrid Classical-Quantum Transfer Learning (QTL)** models. It was developed to support the revision of the manuscript "Hybrid Classical-Quantum Transfer Learning with Noisy Quantum Circuits" (CMES ID: 82712).

The framework provides a unified interface to compare high-performance classical backbones (ResNet, MobileNet, etc.) combined with Variational Quantum Circuit (VQC) heads simulated via **PennyLane** and **Qiskit**, inclusive of realistic noise modeling based on **IBM Heron r2** hardware.

## Key Features
- **Fair Baseline System:** Includes parameter-matched MLP heads to ensure quantum-classical comparisons are scientifically sound.
- **Robust SLURM Orchestration:** Interactive `manager.py` tool for managing 692+ experiments on high-performance clusters (Hercules).
- **Advanced Noise Modeling:** Integrated IBM hardware calibration (T1/T2 times, gate errors, readout noise).
- **Statistical Significance:** Automatic execution across 5 seeds with $\mu \pm \sigma$ reporting in LaTeX.
- **Sustainability Tracking:** Integrated **CodeCarbon** for tracking energy footprints (kWh) of quantum simulations.
- **Comprehensive Visualization:** Per-epoch learning curves (Loss/Acc), ROC-AUC, PR-curves, and confusion matrices.

## Quick Start (Hercules Cluster)
1. **Clone and Install:**
   ```bash
   git clone <repo-url>
   cd qtl_experiments
   pip install -r requirements.txt
   ```
2. **Launch Management HUB:**
   ```bash
   python manager.py
   # OR
   bash hercules_orchestrator.sh
   ```
3. **Common Workflow:**
   - Use `[R]` to refresh command lists.
   - Use `[F]` to launch the Full Pipeline on SLURM.
   - Use `[M]` to monitor progress and completion rates.
   - Use `[T]` to generate LaTeX tables once the jobs finish.

## Project Structure
- `data/`: Dataset loaders and augmentation logic.
- `heads/`: Implementation of Classical, PennyLane, and Qiskit heads.
- `results/`: Output directory for models, checkpoints, and CSV logs.
- `paper/`: Automatically generated LaTeX tables and methodology documents.
- `manager.py`: Professional CLI for cluster execution.
- `runner.py`: Core logic for hyperparameter sweep orchestration.
- `trainer.py`: Robust training loop with epoch-level checkpointing and resume support.

## Authors & Citation
*Initial submission revision for CMES (April 2026).*
For questions, contact the primary author at the institutional email provided in the manuscript.

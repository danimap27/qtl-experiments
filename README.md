# QTL Experimentation Framework

**Hybrid Classical-Quantum Transfer Learning with Noisy Quantum Circuits**  
Revision of article CMES-82712 · CMES Journal · April 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
   - [Local / Generic Linux](#local--generic-linux)
   - [HERCULES Cluster (CICA, SLURM)](#hercules-cluster-cica-slurm)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Running Experiments](#running-experiments)
   - [Smoke Test (1 epoch)](#smoke-test-1-epoch)
   - [Main Battery (660 runs)](#main-battery-660-runs)
   - [Additional Studies](#additional-studies)
   - [SLURM Job Arrays](#slurm-job-arrays)
7. [Monitoring Progress](#monitoring-progress)
8. [Collecting Results](#collecting-results)
9. [Visualization](#visualization)
10. [Tabular & Clustering Tasks](#tabular--clustering-tasks)
11. [Checkpoints & Resume](#checkpoints--resume)
12. [Noise Model](#noise-model)
13. [Statistical Protocol](#statistical-protocol)
14. [Troubleshooting](#troubleshooting)

---

## Overview

This framework orchestrates **660 reproducible experiments** comparing:

| Dimension | Values |
|-----------|--------|
| Datasets | hymenoptera, brain_tumor, cats_vs_dogs, solar_dust |
| Backbones | ResNet-18, MobileNetV2, EfficientNet-B0, RegNet-X-400MF |
| Heads | linear, mlp\_a, mlp\_b, pl\_ideal, pl\_noisy, qk\_ideal, qk\_noisy |
| Seeds | 0, 42, 123, 456, 789 |

Additional studies included: qubit/depth ablation `[4,8,16] × [1,3,5]`, noise channel decomposition (6 isolated channels), barren plateau analysis, and transpilation scalability.

The noise model is calibrated to **IBM Heron r2** specs (T1=250µs, T2=150µs, p1q=0.0002, p2q=0.005, readout=0.012). **All experiments run on classical simulators — no real QPU is required or used.**

---

## Requirements

- **Python 3.11+** (Python 3.8 is NOT compatible — see [Troubleshooting](#troubleshooting))
- CPU-only machine or GPU (CUDA optional, not required)
- ~4 GB RAM minimum; 8 GB recommended for noisy quantum heads

---

## Installation

### Local / Generic Linux

```bash
# 1. Clone the repository
git clone https://github.com/danimap27/qtl_experiments.git
cd qtl_experiments

# 2. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `tqdm` and `matplotlib` are pulled in transitively. If you want progress
> bars explicitly: `pip install tqdm matplotlib`.

### HERCULES Cluster (CICA, SLURM)

```bash
# 1. Load Python 3.11 module (REQUIRED — default Python 3.8 is incompatible)
module load Python/3.11.3-GCCcore-12.3.0

# 2. Create virtual environment in home directory
python3 -m venv ~/qtl_venv311
source ~/qtl_venv311/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import pennylane, qiskit, torch; print('All imports OK')"
```

> The SLURM scripts (`hercules_dani.sh`, `hercules_david.sh`) already include the
> `module load` and `source activate` lines automatically.

---

## Project Structure

```
qtl_experiments/
├── runner.py               # Main orchestrator — generates and runs all experiments
├── trainer.py              # PyTorch training loop with tqdm, checkpoints, metrics
├── visualization.py        # 13 plot functions (learning curves, ROC, clustering, …)
├── config.yaml             # Master configuration file
├── merge_results.py        # Consolidate partial CSVs from multiple SLURM nodes
├── bp_analysis.py          # Barren plateau analysis (gradient variance vs. qubits)
├── transpilation.py        # Circuit depth analysis for IBM Torino
├── requirements.txt        # Pinned dependencies
│
├── data/
│   ├── loader.py           # Unified loader: dispatches image ↔ tabular
│   ├── tabular_loader.py   # CSV / NumPy tabular dataset support
│   └── download_datasets.py
│
├── heads/
│   ├── __init__.py         # Factory: get_head(config, feature_dim, num_classes)
│   ├── linear_head.py      # Linear baseline
│   ├── mlp_a_head.py       # Parameter-matched MLP (~12 params) ← key contribution
│   ├── mlp_b_head.py       # Standard over-parameterised MLP
│   ├── pennylane_head.py   # VQC via PennyLane (ideal + noisy)
│   ├── qiskit_head.py      # VQC via Qiskit 1.x V2 API (ideal + noisy)
│   └── clustering_head.py  # KMeansHead / DBSCANHead for unsupervised tasks
│
├── results/                # Auto-created. CSV outputs + plots per run
├── hercules_dani.sh        # SLURM job array script
├── hercules_david.sh       # SLURM job array script (alternate node config)
└── hercules_bp.sh          # SLURM script for barren plateau study
```

---

## Configuration

All experiments are controlled by `config.yaml`. Key sections:

```yaml
seeds: [0, 42, 123, 456, 789]   # 5 seeds for statistical validation

datasets:
  - name: "hymenoptera"
    path: "./data/datasets/hymenoptera"
    num_classes: 2
    image_size: 224

training:
  optimizer: "adam"
  lr: 0.001
  batch_size: 16
  epochs: 10

checkpoints:
  save_best: true      # saves ckpt_best.pt
  save_last: true      # saves ckpt_last.pt
  resume: false        # set true to resume from ckpt_best.pt

visualization:
  enabled: true
  dpi: 150
```

For tabular datasets, add `dataset_type: "tabular"` to the dataset block.  
For clustering, add `type: "clustering"` and `algorithm: "kmeans"` to the head block.

---

## Running Experiments

### Smoke Test (1 epoch)

Before launching the full battery, verify the installation works end-to-end.

**Step 1 — Create a test config:**

```bash
cp config.yaml config_test.yaml
# Edit config_test.yaml: set epochs: 1 and seeds: [42]
```

**Step 2 — Test the simplest case first:**

```bash
python runner.py --config config_test.yaml \
  --dataset hymenoptera --backbone resnet18 --head linear \
  --machine-id test
```

**Step 3 — Test a quantum head:**

```bash
python runner.py --config config_test.yaml \
  --dataset hymenoptera --backbone resnet18 --head pl_ideal \
  --machine-id test
```

**Step 4 — Run all 112 smoke-test runs (4×4×7×1 seed):**

```bash
python runner.py --config config_test.yaml --parallel 4 --machine-id smoke
```

Check for errors:

```bash
cat results/errors.log      # should be empty
ls results/*.csv | wc -l    # should equal number of completed runs
```

### Main Battery (660 runs)

```bash
# Dry run — count runs without executing
python runner.py --config config.yaml --dry-run --count

# Run locally with 4 parallel workers
python runner.py --config config.yaml --parallel 4 --machine-id local

# Filter by specific dimensions
python runner.py --config config.yaml --head-type quantum --parallel 2
python runner.py --config config.yaml --dataset hymenoptera --backbone resnet18
python runner.py --config config.yaml --seed 42,123
```

### Additional Studies

```bash
# Qubit/depth ablation [4,8,16] × [1,3,5]
python runner.py --config config.yaml --study ablation --parallel 4

# Noise channel decomposition (6 isolated channels)
python runner.py --config config.yaml --study noise_decomposition --parallel 2

# Sim-as-hardware (reduced shots=100)
python runner.py --config config.yaml --study sim_as_hardware

# Barren plateau analysis (standalone script)
python bp_analysis.py --config config.yaml

# Transpilation scalability
python transpilation.py --config config.yaml
```

### SLURM Job Arrays

```bash
# 1. Export one command per run
python runner.py --config config.yaml --dry-run --export-commands > cmds.sh
wc -l cmds.sh    # verify run count

# 2. Submit job array
sbatch hercules_dani.sh

# 3. Monitor
squeue -u $USER
tail -f results/errors.log
```

---

## Monitoring Progress

During execution, each run prints:

```
Epoch   1/10 [████████░░░░░░░░░░░░]  10% | Loss 0.6821 | Val 0.6543 | Acc 58.3% | lr 1.00e-03 | 0:00:12
  Epoch   1 |  0.6821 |  0.6543 |  58.3% |  61.1% | 1.0e-03 | 0:12 |
```

- **Per-batch bar** — loss, accuracy, LR, ETA for current epoch
- **Epoch summary line** — train/val loss & accuracy, LR, elapsed time
- `[CKPT✓]` flag — printed when a new best checkpoint is saved

---

## Collecting Results

Each completed run saves a `{run_id}.csv` in `results/`. To consolidate:

```bash
python merge_results.py results/ --output results_final.csv

# Quick summary
python -c "
import pandas as pd
df = pd.read_csv('results_final.csv')
print(df.groupby('head')['test_acc'].agg(['mean','std']).round(4))
"
```

---

## Visualization

Plots are generated automatically after each run and saved to `results/plots/{run_id}/`.

| Plot | File |
|------|------|
| Learning curves (loss + acc) | `learning_curves.png` |
| Confusion matrix (raw + normalised) | `confusion_matrix.png` |
| ROC curve with AUC | `roc_curve.png` |
| Precision-Recall curve | `pr_curve.png` |
| Per-class metrics bars | `class_metrics.png` |
| Summary panel (2×3) | `classification_summary.png` |
| Cluster scatter PCA / t-SNE | `cluster_scatter.png` |
| Silhouette analysis | `silhouette.png` |
| Elbow curve (KMeans) | `elbow_curve.png` |

To regenerate plots for an existing run:

```python
from visualization import generate_all_plots
generate_all_plots(history, y_true, y_pred, probs, run_id="my_run", output_dir="results")
```

---

## Tabular & Clustering Tasks

### Tabular datasets

Add `dataset_type` to the dataset config:

```yaml
datasets:
  - name: "my_tabular"
    path: "./data/datasets/my_tabular"   # can be a .csv file or a directory
    num_classes: 3
    dataset_type: "tabular"              # activates tabular_loader.py
    label_column: "label"               # column name for class labels
    scale: true                         # StandardScaler (fit on train only)
```

Supported formats (auto-detected):

| Format | Detection condition |
|--------|---------------------|
| NumPy pre-split | `X_train.npy` + `y_train.npy` exist in directory |
| CSV pre-split | `train.csv` exists in directory |
| Single CSV | path points to a `.csv` file or `data.csv` exists |

### Clustering heads

```yaml
heads:
  - name: "cluster_kmeans"
    type: "clustering"
    task_type: "clustering"
    algorithm: "kmeans"    # or "dbscan"
    n_clusters: 3
```

For unsupervised mode (no labels), set `num_classes: 0` in the dataset config.

---

## Checkpoints & Resume

Checkpoints are saved to `results/{run_id}/`:

| File | Contents |
|------|----------|
| `ckpt_best.pt` | Best validation accuracy checkpoint |
| `ckpt_last.pt` | Last completed epoch |
| `ckpt_epochNNN.pt` | Per-epoch (if `save_every_n_epochs > 0`) |

To resume an interrupted run, set `resume: true` in `config.yaml`. The trainer will
automatically load `ckpt_best.pt` and continue from the saved epoch.

---

## Noise Model

The IBM Heron r2 noise model is implemented via two independent backends:

**PennyLane** (`pl_noisy`): Kraus operator channels inserted after every gate.

```
Amplitude Damping:  γ = 1 − exp(−t₁q / T₁)   →  qml.AmplitudeDamping
Phase Damping:      λ = 1 − exp(−t₂q / T₂)   →  qml.PhaseDamping
Depolarizing:       p = p₁q (1q) / p₂q (2q)  →  qml.DepolarizingChannel
```

**Qiskit** (`qk_noisy`): `qiskit_aer.noise.NoiseModel` passed to `AerEstimatorV2`.

```
Gate error:    depolarizing_error(p₁q, 1) on all 1-qubit gates
               depolarizing_error(p₂q, 2) on all 2-qubit gates
Readout error: ReadoutError([[1-p_ro, p_ro], [p_ro, 1-p_ro]])
Shots:         1024 (configurable via shots: in the head config)
```

> **Calibration values (IBM Heron r2):**
> T₁ = 250 µs · T₂ = 150 µs · t₁q = 32 ns · t₂q = 68 ns
> p₁q = 0.0002 · p₂q = 0.005 · p_readout = 0.012

---

## Statistical Protocol

All results are validated with the following pipeline:

1. **5 independent seeds** per configuration: `[0, 42, 123, 456, 789]`
2. **Shapiro-Wilk** normality test on accuracy distributions across seeds
3. **Welch t-test** (if normal) or **Mann-Whitney U** (if non-normal) for pairwise comparisons
4. **Bonferroni correction** for multiple comparisons (α = 0.05)
5. Significance levels reported as: `*` p<0.05 · `**` p<0.01 · `***` p<0.001

---

## Troubleshooting

**`TypeError: 'type' object is not subscriptable` (pennylane/queuing.py)**

PennyLane ≥ 0.38 uses Python 3.9+ type hint syntax. Fix: use Python 3.11.

```bash
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv ~/qtl_venv311 && source ~/qtl_venv311/bin/activate
pip install -r requirements.txt
```

**`ImportError: cannot import name 'Estimator' from 'qiskit.primitives'`**

Qiskit 1.x removed V1 primitives. The framework already uses V2 (`StatevectorEstimator`,
`AerEstimatorV2`). Ensure `qiskit==1.2.4` is installed — do not upgrade beyond 1.2.x
with Python 3.8 environments.

**`fatal: not a git repository`**

The `.git` directory lives inside `qtl_experiments/`. Make sure you `cd qtl_experiments`
before running git commands, and set:

```bash
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1
```

**Runs complete but `results/errors.log` has entries**

Check the specific run ID listed in the log:

```bash
python runner.py --config config.yaml \
  --dataset <dataset> --backbone <backbone> --head <head> \
  --verbose 2 --machine-id debug
```

`--verbose 2` enables full tracebacks in the console.

**`qk_noisy` is very slow**

Reduce shots for testing:

```yaml
# config_test.yaml
heads:
  - name: "qk_noisy"
    shots: 256   # default is 1024
```

---

## License

Academic use only. Code developed for the revision of manuscript CMES-82712.

---

*Generated by the QTL Experimentation Framework · Daniel Martínez Pérez · April 2026*

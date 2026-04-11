"""
Tabular dataset loader for non-image experiments.

Supports CSV files and NumPy arrays with automatic preprocessing:
- StandardScaler normalization
- Train / val / test splits (explicit files or ratio-based)
- Optional label column or unsupervised (no labels) mode
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_tabular_dataset(
    dataset_config: Dict,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load a tabular (CSV / NumPy) dataset and return train/val/test DataLoaders.

    Supported directory structures
    --------------------------------
    Structure A — single CSV file with label column:
        dataset_path/data.csv          (or the path itself is the CSV)

    Structure B — pre-split CSV files:
        dataset_path/train.csv
        dataset_path/val.csv           (optional)
        dataset_path/test.csv          (optional)

    Structure C — NumPy arrays:
        dataset_path/X_train.npy  + dataset_path/y_train.npy
        dataset_path/X_val.npy    + dataset_path/y_val.npy    (optional)
        dataset_path/X_test.npy   + dataset_path/y_test.npy   (optional)

    dataset_config keys
    -------------------
    Required:
        name        (str)  : Dataset identifier
        path        (str)  : Path to CSV file or directory
        num_classes (int)  : Number of classes (0 for unsupervised)

    Optional:
        label_column  (str,  default 'label')  : Column name for labels in CSV
        feature_cols  (list, default None)     : Column names to use as features;
                                                 if None, all non-label columns are used
        scale         (bool, default True)     : Apply StandardScaler normalization
        train_ratio   (float, default 0.7)     : Train fraction for ratio-based split
        val_ratio     (float, default 0.15)    : Val fraction for ratio-based split
        # (remainder becomes test)

    Returns
    -------
    Tuple of (train_loader, val_loader, test_loader)
    """
    required = {"name", "path", "num_classes"}
    if not required.issubset(dataset_config.keys()):
        raise ValueError(
            f"dataset_config must contain {required}. Got: {set(dataset_config.keys())}"
        )

    name           = dataset_config["name"]
    path           = Path(dataset_config["path"])
    num_classes    = dataset_config["num_classes"]
    label_col      = dataset_config.get("label_column", "label")
    feature_cols   = dataset_config.get("feature_cols", None)
    scale          = dataset_config.get("scale", True)
    train_ratio    = dataset_config.get("train_ratio", 0.70)
    val_ratio      = dataset_config.get("val_ratio",   0.15)
    supervised     = num_classes > 0

    logger.info(f"Loading tabular dataset '{name}' from {path}")

    # ------------------------------------------------------------------ #
    # 1. Load raw arrays                                                   #
    # ------------------------------------------------------------------ #
    X_train, y_train, X_val, y_val, X_test, y_test = _load_arrays(
        path, label_col, feature_cols, supervised, train_ratio, val_ratio, seed
    )

    # ------------------------------------------------------------------ #
    # 2. Normalize features (fit on train only)                            #
    # ------------------------------------------------------------------ #
    if scale:
        mean  = X_train.mean(axis=0, keepdims=True)
        std   = X_train.std(axis=0, keepdims=True)
        std   = np.where(std == 0, 1.0, std)   # avoid division by zero
        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std
        X_test  = (X_test  - mean) / std
        logger.debug(f"Features scaled: mean={mean.mean():.4f}, std={std.mean():.4f}")

    feature_dim = X_train.shape[1]
    logger.info(
        f"Feature dim: {feature_dim} | "
        f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # ------------------------------------------------------------------ #
    # 3. Build TensorDatasets                                              #
    # ------------------------------------------------------------------ #
    def _make_dataset(X, y):
        X_t = torch.from_numpy(X).float()
        if supervised and y is not None:
            y_t = torch.from_numpy(y).long()
            return TensorDataset(X_t, y_t)
        else:
            # Unsupervised: return (X, dummy_label=0) so DataLoader API stays consistent
            dummy = torch.zeros(len(X_t), dtype=torch.long)
            return TensorDataset(X_t, dummy)

    train_ds = _make_dataset(X_train, y_train)
    val_ds   = _make_dataset(X_val,   y_val)
    test_ds  = _make_dataset(X_test,  y_test)

    # ------------------------------------------------------------------ #
    # 4. DataLoaders                                                       #
    # ------------------------------------------------------------------ #
    loader_kwargs = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    logger.info(
        f"DataLoaders created: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches, {len(test_loader)} test batches"
    )

    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _load_arrays(
    path: Path,
    label_col: str,
    feature_cols: Optional[list],
    supervised: bool,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """
    Detect dataset format and return six numpy arrays:
    X_train, y_train, X_val, y_val, X_test, y_test
    (y_* are None when unsupervised)
    """
    # ---- NumPy format ------------------------------------------------
    if (path / "X_train.npy").exists():
        return _load_numpy_splits(path, supervised)

    # ---- Pre-split CSV format ----------------------------------------
    if (path / "train.csv").exists():
        return _load_csv_splits(path, label_col, feature_cols, supervised,
                                val_ratio, seed)

    # ---- Single CSV file (path is file OR path/data.csv) -------------
    csv_candidates = [path, path / "data.csv"]
    for csv_path in csv_candidates:
        if csv_path.is_file() and csv_path.suffix == ".csv":
            return _load_single_csv(csv_path, label_col, feature_cols, supervised,
                                    train_ratio, val_ratio, seed)

    raise FileNotFoundError(
        f"Cannot find tabular data at '{path}'. "
        "Expected one of: X_train.npy, train.csv, data.csv, or a .csv file."
    )


def _load_numpy_splits(path: Path, supervised: bool):
    """Load pre-split NumPy arrays."""
    X_train = np.load(path / "X_train.npy")
    X_val   = np.load(path / "X_val.npy")   if (path / "X_val.npy").exists()  else None
    X_test  = np.load(path / "X_test.npy")  if (path / "X_test.npy").exists() else None

    y_train = y_val = y_test = None
    if supervised:
        y_train = np.load(path / "y_train.npy").astype(np.int64)
        if (path / "y_val.npy").exists():
            y_val  = np.load(path / "y_val.npy").astype(np.int64)
        if (path / "y_test.npy").exists():
            y_test = np.load(path / "y_test.npy").astype(np.int64)

    # Fallback: if val/test missing, reuse test/val respectively
    if X_val is None and X_test is not None:
        X_val, y_val = X_test, y_test
    elif X_test is None and X_val is not None:
        X_test, y_test = X_val, y_val
    elif X_val is None and X_test is None:
        # Only train available — split 80/20
        X_train, y_train, X_val, y_val = _ratio_split(X_train, y_train, 0.8, 42)
        X_test, y_test = X_val, y_val

    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_csv_splits(
    path: Path, label_col: str, feature_cols, supervised: bool,
    val_ratio: float, seed: int
):
    """Load from pre-split train.csv [/ val.csv / test.csv]."""
    import csv as _csv_mod

    def _read(csv_path):
        return _csv_to_arrays(csv_path, label_col, feature_cols, supervised)

    X_train, y_train = _read(path / "train.csv")

    if (path / "val.csv").exists():
        X_val, y_val = _read(path / "val.csv")
    else:
        X_train, y_train, X_val, y_val = _ratio_split(
            X_train, y_train, 1.0 - val_ratio, seed
        )

    if (path / "test.csv").exists():
        X_test, y_test = _read(path / "test.csv")
    else:
        X_test, y_test = X_val, y_val

    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_single_csv(
    csv_path: Path, label_col: str, feature_cols, supervised: bool,
    train_ratio: float, val_ratio: float, seed: int
):
    """Load a single CSV and perform ratio-based splits."""
    X_all, y_all = _csv_to_arrays(csv_path, label_col, feature_cols, supervised)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0"
        )

    # Split: train | (val + test)
    X_train, y_train, X_rest, y_rest = _ratio_split(
        X_all, y_all, train_ratio, seed
    )

    if test_ratio > 0 and len(X_rest) > 1:
        val_frac = val_ratio / (val_ratio + test_ratio)
        X_val, y_val, X_test, y_test = _ratio_split(X_rest, y_rest, val_frac, seed + 1)
    else:
        X_val, y_val   = X_rest, y_rest
        X_test, y_test = X_rest, y_rest

    return X_train, y_train, X_val, y_val, X_test, y_test


def _csv_to_arrays(csv_path: Path, label_col: str, feature_cols, supervised: bool):
    """
    Parse a CSV file into numpy feature matrix X and optional label vector y.
    Handles missing/non-numeric gracefully by dropping bad rows.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Select feature columns
        if feature_cols:
            X_cols = [c for c in feature_cols if c in df.columns]
        else:
            X_cols = [c for c in df.columns if c != label_col]

        X = df[X_cols].values.astype(np.float32)

        y = None
        if supervised and label_col in df.columns:
            raw_labels = df[label_col].values
            # Encode string labels to integers if needed
            if raw_labels.dtype.kind in ("U", "S", "O"):
                unique = np.unique(raw_labels)
                label_map = {v: i for i, v in enumerate(unique)}
                raw_labels = np.array([label_map[v] for v in raw_labels])
            y = raw_labels.astype(np.int64)

        return X, y

    except ImportError:
        # Fallback: numpy-based CSV parsing (no pandas)
        return _csv_to_arrays_numpy(csv_path, label_col, feature_cols, supervised)


def _csv_to_arrays_numpy(csv_path: Path, label_col: str, feature_cols, supervised: bool):
    """Minimal CSV parser using only numpy (no pandas dependency)."""
    with open(csv_path, "r", newline="") as f:
        lines = f.readlines()

    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        rows.append([v.strip() for v in line.split(",")])

    data = np.array(rows)

    if feature_cols:
        feat_idx = [header.index(c) for c in feature_cols if c in header]
    else:
        feat_idx = [i for i, h in enumerate(header) if h != label_col]

    X = data[:, feat_idx].astype(np.float32)

    y = None
    if supervised and label_col in header:
        lbl_idx = header.index(label_col)
        raw = data[:, lbl_idx]
        try:
            y = raw.astype(np.int64)
        except ValueError:
            unique = np.unique(raw)
            label_map = {v: i for i, v in enumerate(unique)}
            y = np.array([label_map[v] for v in raw], dtype=np.int64)

    return X, y


def _ratio_split(X, y, train_frac: float, seed: int):
    """
    Split arrays X (and optionally y) into two parts by train_frac ratio.
    Returns X_a, y_a, X_b, y_b  (y_* may be None).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    n_train = max(1, int(n * train_frac))

    idx = rng.permutation(n)
    idx_a, idx_b = idx[:n_train], idx[n_train:]

    X_a, X_b = X[idx_a], X[idx_b]
    y_a = y[idx_a] if y is not None else None
    y_b = y[idx_b] if y is not None else None

    return X_a, y_a, X_b, y_b

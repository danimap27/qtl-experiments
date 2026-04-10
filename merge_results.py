#!/usr/bin/env python3
"""
merge_results.py — Consolidates CSVs from multiple machines.

After parallel execution across lab PCs and HERCULES, each machine
produces its own CSV files (runs_pc1.csv, runs_hercules_dani.csv, etc.).
This script merges them into unified files for analysis.

Usage:
    python merge_results.py --results-dir ./results
    python merge_results.py --results-dir ./results --dry-run
    python merge_results.py --results-dir ./results --verify
"""

import pandas as pd
import glob
import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def find_partial_files(results_dir: str, prefix: str) -> list:
    """Find all partial CSV files matching prefix_*.csv pattern."""
    pattern = str(Path(results_dir) / f"{prefix}_*.csv")
    return sorted(glob.glob(pattern))


def merge_csv_files(files: list, output_path: str,
                    id_column: str = "run_id") -> pd.DataFrame | None:
    """
    Concatenate multiple partial CSVs into a single output file.

    - Exact-duplicate rows (same id_column, same data): deduplicated with warning.
    - Conflicting rows (same id_column, different data): hard error.
    """
    if not files:
        logger.info(f"  No files found for {output_path}")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        logger.info(f"  {Path(f).name}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # ---- duplicate detection ----
    if id_column in merged.columns:
        dupes = merged[merged.duplicated(subset=[id_column], keep=False)]
        if len(dupes) > 0:
            exact_dupes = merged[merged.duplicated(keep=False)]
            if len(exact_dupes) == len(dupes):
                n_removed = len(dupes) // 2
                logger.warning(f"  {n_removed} exact duplicates removed")
                merged = merged.drop_duplicates(subset=[id_column], keep="first")
            else:
                logger.error(
                    f"  {len(dupes)} rows share run_id but differ in data — "
                    "cannot merge automatically"
                )
                for rid in dupes[id_column].unique():
                    rows = dupes[dupes[id_column] == rid]
                    logger.error(f"    Conflicting run_id: {rid}")
                    for _, row in rows.iterrows():
                        logger.error(f"      {row.to_dict()}")
                sys.exit(1)

    # ---- sort ----
    if "timestamp" in merged.columns:
        merged = merged.sort_values("timestamp").reset_index(drop=True)

    merged.to_csv(output_path, index=False)
    logger.info(f"  -> {output_path}: {len(merged)} rows")
    return merged


# ------------------------------------------------------------------ #
#  Verification                                                       #
# ------------------------------------------------------------------ #

def verify_merge(results_dir: str) -> int:
    """
    Post-merge integrity checks.

    Returns the number of errors found (0 = clean merge).
    """
    rd = Path(results_dir)
    errors = 0

    # --- runs.csv ---
    runs_path = rd / "runs.csv"
    if not runs_path.exists():
        logger.error("[FAIL] runs.csv not found")
        return 1
    runs = pd.read_csv(runs_path)

    dup = runs[runs.duplicated(subset=["run_id"])]
    if len(dup) > 0:
        logger.error(f"[FAIL] runs.csv: {len(dup)} duplicate run_ids")
        errors += 1
    else:
        logger.info(f"[OK] runs.csv: {len(runs)} runs, 0 duplicates")

    # --- predictions.csv ---
    preds_path = rd / "predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        pred_ids = set(preds["run_id"].unique())
        run_ids = set(runs["run_id"].unique())
        orphans = pred_ids - run_ids
        if orphans:
            logger.error(
                f"[FAIL] predictions.csv: {len(orphans)} orphan run_ids "
                f"(not in runs.csv)"
            )
            errors += 1
        else:
            logger.info("[OK] predictions.csv: all run_ids present in runs.csv")

        # probability sanity
        if "y_prob_0" in preds.columns and "y_prob_1" in preds.columns:
            prob_sum = preds["y_prob_0"] + preds["y_prob_1"]
            bad = (prob_sum - 1.0).abs() > 0.01
            if bad.any():
                logger.error(
                    f"[FAIL] predictions.csv: {bad.sum()} rows with "
                    "probabilities that don't sum to 1"
                )
                errors += 1
            else:
                logger.info("[OK] predictions.csv: probabilities consistent")
    else:
        logger.warning("[WARN] predictions.csv not found — skipping checks")

    # --- training_log.csv ---
    log_path = rd / "training_log.csv"
    if log_path.exists():
        logs = pd.read_csv(log_path)
        log_ids = set(logs["run_id"].unique())
        run_ids = set(runs["run_id"].unique())
        orphans = log_ids - run_ids
        if orphans:
            logger.error(
                f"[FAIL] training_log.csv: {len(orphans)} orphan run_ids"
            )
            errors += 1
        else:
            logger.info("[OK] training_log.csv: all run_ids present in runs.csv")
    else:
        logger.warning("[WARN] training_log.csv not found — skipping checks")

    # --- NaN check in runs ---
    critical_cols = [
        "test_accuracy", "test_precision", "test_recall", "test_f1",
    ]
    for col in critical_cols:
        if col in runs.columns:
            n_nan = runs[col].isna().sum()
            if n_nan > 0:
                logger.warning(f"[WARN] runs.csv: {n_nan} NaN values in {col}")

    # --- summary ---
    if errors == 0:
        logger.info(f"\n[RESULT] Merge valid: {len(runs)} runs, 0 errors")
    else:
        logger.error(f"\n[RESULT] Merge has {errors} errors")
    return errors


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate CSVs from multiple machines"
    )
    parser.add_argument(
        "--results-dir", default="./results",
        help="Directory containing partial CSV files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show merge plan without executing",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run integrity checks on already-merged files",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.verify:
        sys.exit(verify_merge(results_dir))

    # ---- discover partial files ----
    file_groups = {
        "runs":                   find_partial_files(results_dir, "runs"),
        "predictions":            find_partial_files(results_dir, "predictions"),
        "training_log":           find_partial_files(results_dir, "training_log"),
        "bp_results":             find_partial_files(results_dir, "bp_results"),
        "transpilation_results":  find_partial_files(results_dir, "transpilation_results"),
    }

    if args.dry_run:
        print("=== DRY RUN ===")
        total_files = 0
        for name, files in file_groups.items():
            print(f"\n{name}:")
            if not files:
                print("  (no files found)")
                continue
            for f in files:
                df = pd.read_csv(f)
                print(f"  {Path(f).name}: {len(df)} rows")
                total_files += 1
        print(f"\nTotal partial files: {total_files}")
        return

    # ---- merge each group ----
    for name, files in file_groups.items():
        if not files:
            continue
        print(f"\nMerging {name}:")
        id_col = "run_id"
        if name == "bp_results":
            id_col = "head"  # no unique run_id in BP results
        elif name == "transpilation_results":
            id_col = "n_qubits"
        merge_csv_files(
            files,
            str(results_dir / f"{name}.csv"),
            id_column=id_col,
        )

    # ---- auto-verify ----
    print("\n=== Post-merge verification ===")
    verify_merge(results_dir)


if __name__ == "__main__":
    main()

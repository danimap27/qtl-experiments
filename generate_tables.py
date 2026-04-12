#!/usr/bin/env python3
"""
generate_tables.py — Automatic LaTeX table generator for the QTL paper.

Reads all runs.csv files from the numbered folders in results/ and generates:
  1. Main table: Accuracy, F1, AUC, and Training Time (mean ± std, 5 seeds).
  2. Ablation table: Qubits × Depth impact.
  3. Efficiency table: Energy consumption (kWh) per model.
  4. Noise decomposition table: Impact of individual noise channels.

Usage:
    python generate_tables.py                         # Use ./results as default
    python generate_tables.py --results-dir ./results --out-dir ./paper/tables
    python generate_tables.py --study ablation
"""

import argparse
import os
import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Formatting labels for the paper
# ──────────────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "linear":       r"\textsc{Linear}",
    "mlp_a":        r"\textsc{MLP-SM}",           # Small / parameter-matched
    "mlp_b":        r"\textsc{MLP-LG}",           # Large standard
    "pl_ideal":     r"\textsc{PL-Ideal}",
    "pl_noisy":     r"\textsc{PL-Noisy}",
    "qk_ideal":     r"\textsc{QK-Ideal}",
    "qk_noisy":     r"\textsc{QK-Noisy}",
}

DATASET_LABELS = {
    "hymenoptera":  "Hymenoptera",
    "brain_tumor":  "Brain Tumor",
    "cats_vs_dogs": r"Cats vs.\ Dogs",
    "solar_dust":   "Solar Dust",
}

BACKBONE_LABELS = {
    "resnet18":       r"ResNet-18",
    "mobilenetv2":    r"MobileNet-V2",
    "efficientnet_b0": r"EfficientNet-B0",
    "regnet_x_400mf": r"RegNet-400MF",
}

HEAD_ORDER = ["linear", "mlp_a", "mlp_b", "pl_ideal", "pl_noisy", "qk_ideal", "qk_noisy"]
DATASET_ORDER = ["hymenoptera", "brain_tumor", "cats_vs_dogs", "solar_dust"]
BACKBONE_ORDER = ["resnet18", "mobilenetv2", "efficientnet_b0", "regnet_x_400mf"]


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_runs(results_dir: str) -> pd.DataFrame:
    """Load all runs.csv files from numbered folders and concatenate them."""
    dfs = []
    pattern = os.path.join(results_dir, "[0-9]*", "runs.csv")
    for csv_path in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to read {csv_path}: {e}")

    if not dfs:
        print(f"[ERROR] No runs.csv files found in {results_dir}/NNN_*/")
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)
    # Remove duplicates, keeping the most recent entry
    all_df = all_df.sort_values("timestamp").drop_duplicates(
        subset=["run_id"], keep="last"
    )
    print(f"[OK] Successfully loaded {len(all_df)} unique runs from {len(dfs)} folders.")
    return all_df


def load_training_logs(results_dir: str) -> pd.DataFrame:
    """Load all training_log.csv files for loss curve analysis."""
    dfs = []
    pattern = os.path.join(results_dir, "[0-9]*", "training_log.csv")
    for csv_path in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX Formatting Helpers
# ──────────────────────────────────────────────────────────────────────────────

def pct(val: float) -> str:
    """Format a 0-1 float as a percentage string with 1 decimal place."""
    return f"{val * 100:.1f}"

def bold(s: str) -> str:
    """Wrap string in LaTeX bold command."""
    return r"\textbf{" + s + "}"

def fmt_mean_std(series: pd.Series, as_pct: bool = True, bold_best: bool = False) -> str:
    """Calculates and formats mean ± standard deviation for a series."""
    if series.empty or series.isna().all():
        return "—"
    mean = series.mean()
    std  = series.std(ddof=1) if len(series) > 1 else 0.0
    if as_pct:
        s = f"{mean*100:.1f} \\pm {std*100:.1f}"
    else:
        s = f"{mean:.1f} \\pm {std:.1f}"
    return f"${s}$"

def fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"

def write_table(path: str, content: str):
    """Save the LaTeX table content to a .tex file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] Table saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Table 1 — Main Experimental Results
# ──────────────────────────────────────────────────────────────────────────────

def make_main_results_table(df: pd.DataFrame, out_dir: str):
    """
    Generates a table showing Accuracy and F1-Score per head across datasets and backbones.
    """
    heads_present = [h for h in HEAD_ORDER if h in df["head"].unique()]
    datasets_present = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    backbones_present = [b for b in BACKBONE_ORDER if b in df["backbone"].unique()]

    # Statistics are grouped across seeds
    n_heads = len(heads_present)
    
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main Experimental Results: Test Accuracy (\%) and F1-Score (\%) across datasets, backbones, and heads. "
                 r"Values reported as mean\,$\pm$\,std over 5 random seeds.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{ll" + "cc" * n_heads + "}")
    lines.append(r"\toprule")

    # Header Level 1: Frameworks/Methods
    midrule_items = []
    header1_parts = ["", ""]
    for h in heads_present:
        label = HEAD_LABELS.get(h, h)
        header1_parts.append(r"\multicolumn{2}{c}{" + label + "}")
        midrule_items.append(f"\\cmidrule(lr){{{3 + heads_present.index(h)*2}-{4 + heads_present.index(h)*2}}}")

    lines.append(" & ".join(header1_parts) + r" \\")
    lines.append(" ".join(midrule_items))

    # Header Level 2: Metric names
    header2_parts = [r"\textbf{Dataset}", r"\textbf{Backbone}"]
    for _ in heads_present:
        header2_parts += [r"Acc\,\%", r"F1\,\%"]
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    for dataset in datasets_present:
        ds_df = df[df["dataset"] == dataset]
        first_in_ds = True
        for backbone in backbones_present:
            bb_df = ds_df[ds_df["backbone"] == backbone]
            if bb_df.empty:
                continue

            row_parts = []
            if first_in_ds:
                row_parts.append(DATASET_LABELS.get(dataset, dataset))
                first_in_ds = False
            else:
                row_parts.append("")
            row_parts.append(BACKBONE_LABELS.get(backbone, backbone))

            for head in heads_present:
                head_df = bb_df[bb_df["head"] == head]
                acc_str = fmt_mean_std(head_df["test_accuracy"], as_pct=True)
                f1_str  = fmt_mean_std(head_df["test_f1"], as_pct=True)
                row_parts += [acc_str, f1_str]

            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\midrule")

    # Finalize table structure
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    content = "\n".join(lines) + "\n"
    write_table(os.path.join(out_dir, "tab_main_results.tex"), content)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Table 2 — Detailed Metrics for Specific Backbone/Dataset
# ──────────────────────────────────────────────────────────────────────────────

def make_full_metrics_table(df: pd.DataFrame, out_dir: str,
                             dataset: str = "hymenoptera",
                             backbone: str = "resnet18"):
    sub = df[(df["dataset"] == dataset) & (df["backbone"] == backbone)]
    heads_present = [h for h in HEAD_ORDER if h in sub["head"].unique()]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive Classification Metrics for "
                 + DATASET_LABELS.get(dataset, dataset)
                 + r" using " + BACKBONE_LABELS.get(backbone, backbone)
                 + r". All values represent mean\,$\pm$\,std over 5 seeds.}")
    lines.append(r"\label{tab:full_metrics_" + dataset + "_" + backbone + "}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Acc\,\%} & \textbf{Prec\,\%} & "
                 r"\textbf{Rec\,\%} & \textbf{F1\,\%} & \textbf{AUC} & "
                 r"\textbf{Training Time} \\")
    lines.append(r"\midrule")

    for head in heads_present:
        hdf = sub[sub["head"] == head]
        if hdf.empty:
            continue
        acc  = fmt_mean_std(hdf["test_accuracy"])
        prec = fmt_mean_std(hdf["test_precision"])
        rec  = fmt_mean_std(hdf["test_recall"])
        f1   = fmt_mean_std(hdf["test_f1"])
        auc  = fmt_mean_std(hdf["test_auc"])
        t    = f"{hdf['train_time_s'].mean():.0f}s" if "train_time_s" in hdf else "—"
        label = HEAD_LABELS.get(head, head)
        lines.append(f"{label} & {acc} & {prec} & {rec} & {f1} & {auc} & {t} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    fname = f"tab_full_metrics_{dataset}_{backbone}.tex"
    write_table(os.path.join(out_dir, fname), content)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Table 3 — Qubit and Depth Ablation Study
# ──────────────────────────────────────────────────────────────────────────────

def make_ablation_table(df: pd.DataFrame, out_dir: str):
    ab = df[df["study"] == "ablation"] if "study" in df.columns else pd.DataFrame()
    if ab.empty:
        # Fallback detection via run_id
        ab = df[df["run_id"].str.contains("ablation|qubit", case=False, na=False)]
    if ab.empty:
        print("  [INFO] No ablation study data found; skipping table generation.")
        return ""

    heads_ab = [h for h in ["pl_ideal", "qk_ideal"] if h in ab["head"].unique()]
    qubits = sorted(ab["n_qubits"].dropna().unique().astype(int))
    depths = sorted(ab["depth"].dropna().unique().astype(int))

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Analysis: Test Accuracy (\%) vs.\ Number of Qubits and Circuit Depth "
                 r"(Hymenoptera, ResNet-18, Seed 42).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{cc" + "c" * len(depths) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Framework} & \textbf{Qubits} & "
                 + " & ".join([f"Depth={d}" for d in depths]) + r" \\")
    lines.append(r"\midrule")

    for head in heads_ab:
        hdf = ab[ab["head"] == head]
        label = HEAD_LABELS.get(head, head)
        first = True
        for q in qubits:
            row = [label if first else "", str(q)]
            first = False
            for d in depths:
                cell = hdf[(hdf["n_qubits"] == q) & (hdf["depth"] == d)]
                if cell.empty:
                    row.append("—")
                else:
                    acc = cell["test_accuracy"].mean()
                    row.append(f"${acc*100:.1f}$")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    write_table(os.path.join(out_dir, "tab_ablation.tex"), content)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Table 4 — Computational Efficiency and Energy Consumption
# ──────────────────────────────────────────────────────────────────────────────

def make_efficiency_table(df: pd.DataFrame, out_dir: str):
    if "energy_kwh" not in df.columns or "train_time_s" not in df.columns:
        print("  [INFO] Efficiency metrics (energy/time) missing; skipping table.")
        return ""

    heads_present = [h for h in HEAD_ORDER if h in df["head"].unique()]
    
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of Training Efficiency and Energy Consumption per Model "
                 r"(Reported as aggregated mean\,$\pm$\,std across all experimental configurations).}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Mean Time (s)} & \textbf{Energy Usage (kWh)} \\")
    lines.append(r"\midrule")

    for head in heads_present:
        hdf = df[df["head"] == head]
        if hdf.empty:
            continue
        t_mean = hdf["train_time_s"].mean()
        t_std  = hdf["train_time_s"].std(ddof=1) if len(hdf) > 1 else 0.0
        e_vals = hdf["energy_kwh"].dropna()
        if e_vals.empty:
            e_str = "—"
        else:
            e_mean = e_vals.mean()
            e_std  = e_vals.std(ddof=1) if len(e_vals) > 1 else 0.0
            e_str  = f"${e_mean*1000:.3f} \\pm {e_std*1000:.4f}$~mWh"
        label = HEAD_LABELS.get(head, head)
        lines.append(f"{label} & ${t_mean:.0f} \\pm {t_std:.0f}$ & {e_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    write_table(os.path.join(out_dir, "tab_efficiency.tex"), content)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Table 5 — Individual Noise Channel Impact Analysis
# ──────────────────────────────────────────────────────────────────────────────

def make_noise_decomp_table(df: pd.DataFrame, out_dir: str):
    nd = df[df["study"] == "noise_decomposition"] if "study" in df.columns else pd.DataFrame()
    if nd.empty:
        nd = df[df["run_id"].str.contains("noise_decomp|amplitude|depolar|phase", case=False, na=False)]
    if nd.empty:
        print("  [INFO] Noise decomposition data not found; skipping table.")
        return ""

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Impact of Discrete Quantum Noise Channels on Accuracy. "
                 r"Baseline: Comprehensive \textsc{PL-Noisy} model.}")
    lines.append(r"\label{tab:noise_decomp}")
    lines.append(r"\begin{tabular}{llc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Active Noise Channel} & \textbf{Accuracy\,\%} \\")
    lines.append(r"\midrule")

    channels = ["amplitude_damping", "phase_damping", "depolarizing"]
    ch_labels = {
        "amplitude_damping": "Amplitude Damping",
        "phase_damping":     "Phase Damping",
        "depolarizing":      "Depolarizing",
    }

    datasets_present = [d for d in DATASET_ORDER if d in nd["dataset"].unique()]
    for dataset in datasets_present:
        ddf = nd[nd["dataset"] == dataset]
        first = True
        for ch in channels:
            if "noise_channels" in ddf.columns:
                chdf = ddf[ddf["noise_channels"].str.contains(ch, na=False)]
            else:
                chdf = ddf[ddf["run_id"].str.contains(ch, na=False)]
            
            acc = chdf["test_accuracy"].mean() if not chdf.empty else float("nan")
            acc_str = f"${acc*100:.1f}$" if not np.isnan(acc) else "—"
            ds_label = DATASET_LABELS.get(dataset, dataset) if first else ""
            lines.append(f"{ds_label} & {ch_labels[ch]} & {acc_str} \\\\")
            first = False
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    write_table(os.path.join(out_dir, "tab_noise_decomp.tex"), content)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Table 6 — Compact Statistical Summary for Robustness (Rebuttal Use)
# ──────────────────────────────────────────────────────────────────────────────

def make_statistical_summary(df: pd.DataFrame, out_dir: str):
    """Generates a summary of statistical distribution across all experiments."""
    heads_present = [h for h in HEAD_ORDER if h in df["head"].unique()]
    metric_col = "test_accuracy"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Aggregate Statistical Summary of Model Performance (Accuracy) across all experimental configurations and datasets.}")
    lines.append(r"\label{tab:stat_summary}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & $N$ & \textbf{Mean\,\%} & \textbf{Std\,\%} "
                 r"& \textbf{Min\,\%} & \textbf{Max\,\%} & \textbf{95\% CI} \\")
    lines.append(r"\midrule")

    # Use scipy for confidence intervals if available
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    for head in heads_present:
        hdf = df[df["head"] == head][metric_col].dropna()
        if hdf.empty:
            continue
        n = len(hdf)
        mean = hdf.mean() * 100
        std  = hdf.std(ddof=1) * 100 if n > 1 else 0.0
        mn   = hdf.min() * 100
        mx   = hdf.max() * 100

        if HAS_SCIPY and n > 1:
            ci = scipy_stats.t.interval(0.95, df=n-1, loc=hdf.mean(), scale=scipy_stats.sem(hdf))
            ci_str = f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"
        else:
            ci_str = "—"

        label = HEAD_LABELS.get(head, head)
        lines.append(f"{label} & {n} & ${mean:.1f}$ & ${std:.1f}$ & "
                     f"${mn:.1f}$ & ${mx:.1f}$ & {ci_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    write_table(os.path.join(out_dir, "tab_stat_summary.tex"), content)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Master TeX Include file
# ──────────────────────────────────────────────────────────────────────────────

def make_master_include(out_dir: str, generated_files: list):
    """Creates a master .tex file that inputs all generated tables."""
    lines = [
        "% ============================================================",
        "% AUTO-GENERATED — Performance Tables for the QTL Paper",
        "% Usage: \\input{paper/tables/tables.tex} in your main document.",
        "% ============================================================",
        "",
    ]
    for f in generated_files:
        lines.append(r"\input{" + f.replace("\\", "/") + "}")
    content = "\n".join(lines) + "\n"
    write_table(os.path.join(out_dir, "tables.tex"), content)


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution Flow
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_tables(results_dir: str, out_dir: str):
    print(f"\n{'='*60}")
    print("  QTL Framework: Professional LaTeX Table Generator")
    print(f"  Source directory : {results_dir}")
    print(f"  Target directory : {out_dir}")
    print(f"{'='*60}\n")

    df = load_all_runs(results_dir)
    if df.empty:
        print("[ERROR] No data found. Please ensure experiment results exist in results/NNN_*/ folders.")
        return

    os.makedirs(out_dir, exist_ok=True)
    generated = []

    print("\n[Step 1/6] Generating Main Experimental Results table...")
    make_main_results_table(df, out_dir)
    generated.append(os.path.join(out_dir, "tab_main_results.tex"))

    print("[Step 2/6] Generating Detailed Evaluation metrics (Hymenoptera/ResNet-18)...")
    make_full_metrics_table(df, out_dir, "hymenoptera", "resnet18")
    generated.append(os.path.join(out_dir, "tab_full_metrics_hymenoptera_resnet18.tex"))

    # Optional: Generate detailed tables for any found configuration
    for ds in df["dataset"].unique():
        for bb in df["backbone"].unique():
            sub = df[(df["dataset"] == ds) & (df["backbone"] == bb)]
            if len(sub) > 0 and not (ds == "hymenoptera" and bb == "resnet18"):
                make_full_metrics_table(df, out_dir, ds, bb)

    print("[Step 3/6] Generating Ablation Study (Qubits vs. Depth) results...")
    make_ablation_table(df, out_dir)
    generated.append(os.path.join(out_dir, "tab_ablation.tex"))

    print("[Step 4/6] Generating Efficiency and Energy Consumption analysis...")
    make_efficiency_table(df, out_dir)
    generated.append(os.path.join(out_dir, "tab_efficiency.tex"))

    print("[Step 5/6] Generating Noise Decomposition by individual channels...")
    make_noise_decomp_table(df, out_dir)
    generated.append(os.path.join(out_dir, "tab_noise_decomp.tex"))

    print("[Step 6/6] Generating Comprehensive Statistical summary...")
    make_statistical_summary(df, out_dir)
    generated.append(os.path.join(out_dir, "tab_stat_summary.tex"))

    print("\n[Step OK] Compiling master LaTeX file (tables.tex)...")
    rel_files = [os.path.relpath(f, start=os.path.dirname(out_dir)) for f in generated]
    make_master_include(out_dir, rel_files)

    print(f"\n[SUCCESS] {len(generated)} tables generated in {out_dir}.")
    print("  You can now include them in your document using: \\input{paper/tables/tables.tex}\n")


def main():
    parser = argparse.ArgumentParser(description="Professional evaluation table generator for the QTL framework.")
    parser.add_argument("--results-dir", default="./results",
                        help="Root directory containing experimental results (default: ./results)")
    parser.add_argument("--out-dir", default="./paper/tables",
                        help="Output directory for .tex files (default: ./paper/tables)")
    args = parser.parse_args()
    generate_all_tables(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()

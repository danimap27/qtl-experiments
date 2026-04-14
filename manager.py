#!/usr/bin/env python3
"""
manager.py — QTL Experiment Control Center & SLURM Monitor

This script provides an interactive menu to:
1. Refresh experiment command lists based on config.yaml.
2. Launch experiments in phases (Classical, Ideal, Noisy, Studies).
3. Monitor progress by scanning results/ folders.
4. Generate final LaTeX tables.

Usage:
    python manager.py
"""

import os
import sys
import subprocess
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Attempt to import pandas for better monitoring; fallback to simple counting
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Constants
CONFIG = "config.yaml"
RESULTS_DIR = "./results"
COMMAND_FILES = {
    "1": ("cmds_1_classical.txt", "Phase 1: Classical Baselines"),
    "2": ("cmds_2_ideal.txt",     "Phase 2: Ideal Quantum"),
    "3": ("cmds_3_noisy.txt",     "Phase 3: Noisy Quantum"),
    "4": ("cmds_4_others.txt",    "Phase 4: Ablation & Studies"),
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*70)
    print("      🚀 QTL EXPERIMENT CONTROL CENTER (HERCULES HUB) 🚀")
    print("="*70)

def run_command(cmd: str, capture: bool = False):
    """Run a shell command."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=True)
            return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {cmd}")
        print(f"Error: {e}")
        return None

def refresh_commands():
    """Regenerate the .txt command files using runner.py."""
    print("\n[INFO] Refreshing command lists from config.yaml...")
    
    # Construct commands
    commands = [
        (f"python runner.py --config {CONFIG} --head linear,mlp_a,mlp_b --dry-run --export-commands > cmds_1_classical.txt", "Phase 1"),
        (f"python runner.py --config {CONFIG} --head pl_ideal,qk_ideal --dry-run --export-commands > cmds_2_ideal.txt", "Phase 2"),
        (f"python runner.py --config {CONFIG} --head pl_noisy,qk_noisy --dry-run --export-commands > cmds_3_noisy.txt", "Phase 3"),
        (f"python runner.py --config {CONFIG} --study ablation --dry-run --export-commands > cmds_4_others.txt", "Phase 4 (Ablation)"),
        (f"python runner.py --config {CONFIG} --study noise_decomposition --dry-run --export-commands >> cmds_4_others.txt", "Phase 4 (Noise)"),
        (f"python runner.py --config {CONFIG} --study sim_as_hardware --dry-run --export-commands >> cmds_4_others.txt", "Phase 4 (Scalability)"),
    ]

    for cmd, name in commands:
        print(f"  Generating {name}...", end=" ", flush=True)
        success = run_command(cmd)
        if success:
            print("[OK]")
        else:
            print("[FAILED]")
    
    print("\n[VERIFY] Task counts:")
    for key, (path, name) in COMMAND_FILES.items():
        count = get_slurm_tasks(path)
        print(f"  - {name}: {count} tasks found")

    print("\n[OK] Refresh complete.")
    input("\nPress Enter to return to menu...")

def get_slurm_tasks(file_path: str) -> int:
    """Count lines in a command file with robust encoding detection."""
    if not os.path.exists(file_path):
        return 0
    
    # Try common encodings to handle files from different OSs
    for enc in ["utf-8-sig", "utf-16", "latin1"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = [line for line in f if line.strip()]
                return len(lines)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0

def submit_phase(key: str, dependency_id: Optional[str] = None):
    """Submit a specific phase to SLURM."""
    if key not in COMMAND_FILES:
        return None
    
    file_path, name = COMMAND_FILES[key]
    n_tasks = get_slurm_tasks(file_path)
    
    if n_tasks == 0:
        print(f"\n[WARN] No tasks found in {file_path}. Use option [R] to refresh first.")
        return None

    dep_arg = f"--dependency=afterok:{dependency_id}" if dependency_id else ""
    
    # Construct sbatch command (ensure no spaces in job name)
    job_name = f"QTL_{key}_{name.split(':')[0]}".replace(" ", "_")
    sbatch_cmd = (
        f"sbatch --parsable --job-name={job_name} "
        f"--array=1-{n_tasks}%40 {dep_arg} "
        f"--export=CMD_FILE={file_path} slurm_generic.sh"
    )
    
    print(f"\n[SUBMIT] Submitting {name} ({n_tasks} tasks)...")
    job_id = run_command(sbatch_cmd, capture=True)
    
    if job_id:
        print(f"[SUCCESS] Job ID: {job_id}")
        return job_id
    return None

def show_monitoring():
    """Scan results and show progress."""
    print("\n[MONITOR] Scanning results/ directory...")
    
    pattern = os.path.join(RESULTS_DIR, "[0-9]*", "runs.csv")
    csv_files = glob.glob(pattern)
    
    total_runs_completed = 0
    if HAS_PANDAS and csv_files:
        try:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs)
            total_runs_completed = len(df.drop_duplicates(subset=["run_id"]))
            
            # Breakdown
            print(f"\n--- Statistics ---")
            print(f"Total Unique Runs Completed: {total_runs_completed}")
            if "head" in df.columns:
                counts = df["head"].value_counts()
                for head, count in counts.items():
                    print(f"  - {head:<15}: {count}")
        except Exception as e:
            print(f"[WARN] Failed to analyze CSVs: {e}")
    else:
        # Fallback: just count folders/files
        total_runs_completed = len(csv_files)
        print(f"Runs completed (folders with runs.csv): {total_runs_completed}")

    # Expected totals (Approximate based on 5 seeds)
    # Main benchmark: 560, Ablation: 64, Noise: 48, SimAsHW: 20 => ~692
    expected = 692
    progress = (total_runs_completed / expected) * 100 if expected > 0 else 0
    
    print(f"\nOverall Progress: {progress:.1f}% ({total_runs_completed}/{expected})")
    
    # Active SLURM jobs
    print("\n--- Active SLURM Jobs ---")
    squeue_out = run_command("squeue -u $USER", capture=True)
    if squeue_out:
        print(squeue_out)
    else:
        print("No active jobs found or squeue unavailable.")
    
    input("\nPress Enter to return to menu...")

def launch_full_pipeline():
    """Submit all phases with dependencies."""
    print("\n[PIPELINE] Launching full sequential pipeline...")
    
    j1 = submit_phase("1")
    j2 = submit_phase("2", j1)
    j3 = submit_phase("3", j2)
    j4 = submit_phase("4", j3)
    
    print("\n[OK] All jobs submitted with sequential dependencies.")
    print(f"Sequence: {j1} -> {j2} -> {j3} -> {j4}")
    input("\nPress Enter to return to menu...")

def generate_tables_trigger():
    """Run generate_tables.py."""
    print("\n[TABLES] Generating LaTeX tables...")
    run_command("python generate_tables.py")
    input("\nPress Enter to return to menu...")

def main():
    while True:
        clear_screen()
        print_header()
        print(" [R] Refresh Command Lists from config.yaml")
        print(" [1] Launch Phase 1: Classical Baselines")
        print(" [2] Launch Phase 2: Ideal Quantum (PennyLane/Qiskit)")
        print(" [3] Launch Phase 3: Noisy Quantum  (PennyLane/Qiskit)")
        print(" [4] Launch Phase 4: Studies (Ablation, Noise, Scalability)")
        print(" [F] Launch FULL PIPELINE (All phases with deps)")
        print(" [M] Monitor Progress & SLURM Queue")
        print(" [T] Generate LaTeX Results Tables")
        print(" [X] Exit")
        print("-" * 70)
        
        choice = input("Select an option: ").strip().upper()
        
        if choice == 'R':
            refresh_commands()
        elif choice == '1':
            submit_phase("1")
            input("\nPress Enter...")
        elif choice == '2':
            submit_phase("2")
            input("\nPress Enter...")
        elif choice == '3':
            submit_phase("3")
            input("\nPress Enter...")
        elif choice == '4':
            submit_phase("4")
            input("\nPress Enter...")
        elif choice == 'F':
            launch_full_pipeline()
        elif choice == 'M':
            show_monitoring()
        elif choice == 'T':
            generate_tables_trigger()
        elif choice == 'X':
            print("\nExiting. Good luck with the submission!\n")
            break
        else:
            print("\nInvalid option. Try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()

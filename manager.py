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
from typing import Dict, List, Optional, Set, Tuple

# Attempt to import pandas for better monitoring; fallback to simple counting
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

# Constants
CONFIG = "config.yaml"
RESULTS_DIR = "./results"
COMMAND_FILES = {
    "1": ("cmds_1_classical.txt", "Phase 1: Classical Baselines"),
    "2": ("cmds_2_ideal.txt",     "Phase 2: Ideal Quantum"),
    "3": ("cmds_3_noisy.txt",     "Phase 3: Noisy Quantum"),
    "4": ("cmds_4_others.txt",    "Phase 4: Ablation & Studies"),
}
EXPECTED_RUNS = 692  # Main: 560, Ablation: 64, Noise: 48, SimAsHW: 20


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("="*70)
    print("      QTL EXPERIMENT CONTROL CENTER (HERCULES HUB)")
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


def get_slurm_tasks(file_path: str) -> int:
    """Count lines in a command file with robust encoding detection."""
    if not os.path.exists(file_path):
        return 0
    for enc in ["utf-8-sig", "utf-16", "latin1"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return sum(1 for line in f if line.strip())
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0


def _progress_bar(done: int, total: int, width: int = 40) -> str:
    """Return an ASCII progress bar string."""
    if total == 0:
        return f"[{'?' * width}] ?/?  ?%"
    pct = done / total
    filled = int(pct * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {done}/{total}  {pct*100:.1f}%"


def _scan_progress() -> Tuple[int, Dict[str, int], Optional[object]]:
    """
    Scan results directory.
    Returns (total_completed, counts_by_head, dataframe_or_None).
    """
    pattern = os.path.join(RESULTS_DIR, "[0-9]*", "runs.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        return 0, {}, None
    if HAS_PANDAS:
        try:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["run_id"])
            counts: Dict[str, int] = {}
            if "head" in df.columns:
                counts = df["head"].value_counts().to_dict()
            return len(df), counts, df
        except Exception:
            pass
    return len(csv_files), {}, None


def _kbhit_nonblock() -> bool:
    """Return True if a key has been pressed (Unix only, non-blocking)."""
    if not HAS_TERMIOS:
        return False
    import select
    return select.select([sys.stdin], [], [], 0)[0] != []


def _collect_completed_ids() -> Set[str]:
    """Collect all completed run_ids across all result subfolders."""
    completed: Set[str] = set()
    pattern = os.path.join(RESULTS_DIR, "[0-9]*", "runs.csv")
    for csv_path in glob.glob(pattern):
        try:
            if HAS_PANDAS:
                df = pd.read_csv(csv_path)
                if "run_id" in df.columns:
                    completed.update(df["run_id"].astype(str))
        except Exception:
            pass
    return completed


def _parse_run_id_from_cmd(line: str) -> Optional[str]:
    """
    Reconstruct run_id from a command line like:
      python runner.py --config ... --dataset D --backbone B --head H --seed S [--study ...]
    run_id format: {dataset}_{backbone}_{head}_{seed}  (main)
                   or {dataset}_{backbone}_{head}_q{n}_d{d}_{seed}  (ablation — skip)
    """
    parts = line.split()
    d = b = h = s = None
    study = None
    for i, p in enumerate(parts):
        if p == "--dataset" and i + 1 < len(parts):
            d = parts[i + 1]
        elif p == "--backbone" and i + 1 < len(parts):
            b = parts[i + 1]
        elif p == "--head" and i + 1 < len(parts):
            h = parts[i + 1]
        elif p == "--seed" and i + 1 < len(parts):
            s = parts[i + 1]
        elif p == "--study" and i + 1 < len(parts):
            study = parts[i + 1]
    if d and b and h and s:
        if study == "ablation":
            return None  # ablation run_ids differ; skip
        return f"{d}_{b}_{h}_{s}"
    return None


def refresh_commands():
    """Regenerate the .txt command files using runner.py."""
    print("\n[INFO] Refreshing command lists from config.yaml...")
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
        print("[OK]" if run_command(cmd) else "[FAILED]")

    print("\n[VERIFY] Task counts:")
    for key, (path, name) in COMMAND_FILES.items():
        print(f"  - {name}: {get_slurm_tasks(path)} tasks")

    print("\n[OK] Refresh complete.")
    input("\nPress Enter to return to menu...")


def check_completed(phase_key: Optional[str] = None) -> Optional[str]:
    """
    Show completed/pending runs for a phase (or all).
    Returns "overwrite_all", "skip_all", or None if cancelled.
    """
    completed_ids = _collect_completed_ids()

    if phase_key:
        files_to_check = [COMMAND_FILES[phase_key]] if phase_key in COMMAND_FILES else []
    else:
        files_to_check = list(COMMAND_FILES.values())

    all_run_ids: List[str] = []
    for file_path, _ in files_to_check:
        if not os.path.exists(file_path):
            continue
        for enc in ["utf-8-sig", "utf-16", "latin1"]:
            try:
                with open(file_path, encoding=enc) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        run_id = _parse_run_id_from_cmd(line)
                        if run_id:
                            all_run_ids.append(run_id)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

    if not all_run_ids:
        print("\n  No parseable command files found. Run [R] first.")
        input("\nEnter to return...")
        return None

    done = [r for r in all_run_ids if r in completed_ids]
    pending = [r for r in all_run_ids if r not in completed_ids]

    print(f"\n  Total: {len(all_run_ids)}  |  Done: {len(done)}  |  Pending: {len(pending)}")

    if done:
        print(f"\n  Completed ({len(done)}):")
        for r in done[:20]:
            print(f"    [x] {r}")
        if len(done) > 20:
            print(f"    ... and {len(done) - 20} more")

    if pending:
        print(f"\n  Pending ({len(pending)}):")
        for r in pending[:20]:
            print(f"    [ ] {r}")
        if len(pending) > 20:
            print(f"    ... and {len(pending) - 20} more")

    if not done:
        input("\nAll runs pending. Enter to return...")
        return None

    print("\n  Completed runs found. What to do when submitting?")
    print("  [S] Skip completed   [O] Overwrite all   [C] Cancel")
    while True:
        choice = input("  Choice: ").strip().upper()
        if choice == "S":
            input("\nWill skip completed runs. Enter to return...")
            return "skip_all"
        if choice == "O":
            input("\nWill overwrite completed runs. Enter to return...")
            return "overwrite_all"
        if choice == "C":
            return None
        print("  Enter S, O, or C.")


def submit_phase(key: str, dependency_id: Optional[str] = None, overwrite: bool = False):
    """Submit a specific phase to SLURM."""
    if key not in COMMAND_FILES:
        return None

    file_path, name = COMMAND_FILES[key]
    n_tasks = get_slurm_tasks(file_path)

    if n_tasks == 0:
        print(f"\n[WARN] No tasks found in {file_path}. Use option [R] to refresh first.")
        return None

    dep_arg = f"--dependency=afterok:{dependency_id}" if dependency_id else ""
    job_name = f"QTL_{key}_{name.split(':')[0]}".replace(" ", "_")
    overwrite_flag = "--overwrite" if overwrite else ""
    sbatch_cmd = (
        f"sbatch --parsable --job-name='{job_name}' "
        f"--array=1-{n_tasks}%40 {dep_arg} "
        f"--export=CMD_FILE={file_path},EXTRA_ARGS={overwrite_flag} slurm_generic.sh"
    )

    print(f"\n[SUBMIT] Submitting {name} ({n_tasks} tasks)...")
    job_id = run_command(sbatch_cmd, capture=True)
    if job_id:
        print(f"[SUCCESS] Job ID: {job_id}")
    return job_id


def show_monitoring():
    """Live progress monitor. Refreshes every 2s. Press any key to exit."""
    old_settings = None
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            clear_screen()
            print_header()
            print()

            total, head_counts, df = _scan_progress()
            bar = _progress_bar(total, EXPECTED_RUNS)
            print(f"  Progress  {bar}")
            print()

            if head_counts:
                print("  Completed by head:")
                for head, count in sorted(head_counts.items()):
                    print(f"    {head:<20}: {count}")
                print()

            # Per-phase breakdown
            print("  Phase breakdown:")
            for key, (file_path, name) in COMMAND_FILES.items():
                n_total = get_slurm_tasks(file_path)
                n_done = 0
                if df is not None and HAS_PANDAS and "head" in df.columns:
                    # Infer phase from head names in command file
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            heads_in_phase = set()
                            for line in f:
                                for part in line.split():
                                    if part.startswith("pl_") or part.startswith("qk_") or part in ("linear", "mlp_a", "mlp_b"):
                                        heads_in_phase.add(part)
                        n_done = int(df[df["head"].isin(heads_in_phase)]["run_id"].nunique())
                    except Exception:
                        pass
                pbar = _progress_bar(n_done, n_total, width=20)
                print(f"    [{key}] {name:<40} {pbar}")
            print()

            # SLURM queue
            squeue = run_command(
                "squeue -u $USER --format='%.10i %.9P %.30j %.8T %.10M' 2>/dev/null",
                capture=True,
            )
            if squeue:
                lines = squeue.splitlines()
                active = max(len(lines) - 1, 0)
                print(f"  Active SLURM jobs: {active}")
                for line in lines[:6]:
                    print(f"    {line}")
            else:
                print("  SLURM queue: not available")

            print()
            print("  " + "-" * 60)
            print("  [Press any key to return to the main menu]")

            if HAS_TERMIOS:
                if _kbhit_nonblock():
                    sys.stdin.read(1)
                    break
            else:
                time.sleep(2)
                continue

            time.sleep(2)

    finally:
        if old_settings is not None and HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print()


def launch_full_pipeline(overwrite: bool = False):
    """Submit all phases with sequential dependencies."""
    print("\n[PIPELINE] Launching full sequential pipeline...")
    j1 = submit_phase("1", overwrite=overwrite)
    j2 = submit_phase("2", j1, overwrite=overwrite)
    j3 = submit_phase("3", j2, overwrite=overwrite)
    j4 = submit_phase("4", j3, overwrite=overwrite)
    print("\n[OK] All jobs submitted with sequential dependencies.")
    print(f"Sequence: {j1} -> {j2} -> {j3} -> {j4}")
    input("\nPress Enter to return to menu...")


def generate_tables_trigger():
    """Run generate_tables.py."""
    print("\n[TABLES] Generating LaTeX tables...")
    run_command("python generate_tables.py")
    input("\nPress Enter to return to menu...")


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    while True:
        clear_screen()
        print_header()
        print(" [R] Refresh Command Lists from config.yaml")
        print(" [1] Launch Phase 1: Classical Baselines")
        print(" [2] Launch Phase 2: Ideal Quantum (PennyLane/Qiskit)")
        print(" [3] Launch Phase 3: Noisy Quantum  (PennyLane/Qiskit)")
        print(" [4] Launch Phase 4: Studies (Ablation, Noise, Scalability)")
        print(" [F] Launch FULL PIPELINE (All phases with deps)")
        print(" [M] Monitor Progress (live, refresh every 2s)")
        print(" [C] Check completed / pending runs")
        print(" [T] Generate LaTeX Results Tables")
        print(" [X] Exit")
        print("-" * 70)

        choice = input("Select an option: ").strip().upper()

        if choice == 'R':
            refresh_commands()
        elif choice in ('1', '2', '3', '4'):
            overwrite_mode = check_completed(phase_key=choice)
            overwrite = overwrite_mode == "overwrite_all"
            submit_phase(choice, overwrite=overwrite)
            input("\nPress Enter...")
        elif choice == 'F':
            overwrite_mode = check_completed()
            overwrite = overwrite_mode == "overwrite_all"
            launch_full_pipeline(overwrite=overwrite)
        elif choice == 'M':
            show_monitoring()
        elif choice == 'C':
            check_completed()
            input("\nPress Enter to return to menu...")
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

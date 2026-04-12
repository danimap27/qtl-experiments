#!/usr/bin/env python3
"""
Quantum Transfer Learning Experimentation Framework - Main Orchestrator

This module serves as the entry point for running quantum transfer learning experiments.
It handles:
  - Configuration loading and validation
  - Run generation from cartesian products of hyperparameters
  - CLI-based filtering and study selection
  - Resumability checks for fault tolerance
  - Parallel and sequential execution
  - CSV logging of results, predictions, and training logs
  - Error handling and progress reporting

Usage:
    python runner.py --config config.yaml --machine-id pc1
    python runner.py --config config.yaml --dataset hymenoptera --backbone resnet18 --dry-run
    python runner.py --config config.yaml --study ablation --parallel 4
"""

import argparse
import csv
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any, Set

import pandas as pd
import yaml


# Global lock for thread-safe CSV writing
_csv_lock = Lock()

# Configure logging with UTF-8 to avoid UnicodeEncodeError on Windows (cp1252)
_log_handler = logging.StreamHandler(sys.stdout)
if hasattr(_log_handler.stream, "reconfigure"):
    _log_handler.stream.reconfigure(encoding="utf-8")
elif hasattr(_log_handler, "setStream"):
    import io
    _log_handler.setStream(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))
_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[_log_handler],
)
logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    run_id: str
    dataset: str
    backbone: str
    head: str
    seed: int
    study: str = "main"
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV serialization."""
        return asdict(self)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with all CLI parameters.
    """
    parser = argparse.ArgumentParser(
        description="Quantum Transfer Learning Experimentation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments from config
  python runner.py --config config.yaml --machine-id pc1

  # Dry run with specific filters
  python runner.py --config config.yaml --dataset hymenoptera --backbone resnet18 --dry-run

  # Count runs without executing
  python runner.py --config config.yaml --dry-run --count

  # Export commands for SLURM job arrays
  python runner.py --config config.yaml --dry-run --export-commands > slurm_commands.sh

  # Run ablation study in parallel
  python runner.py --config config.yaml --study ablation --parallel 4 --machine-id gpu_node_1
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help="Machine identifier (suffix for output CSVs)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset(s), comma-separated (e.g., hymenoptera,brain_tumor)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Filter by backbone(s), comma-separated (e.g., resnet18,mobilenetv2)",
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Filter by head(s), comma-separated (e.g., pl_ideal,pl_noisy)",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        help="Filter by head type (classical, pennylane, qiskit)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Filter by environment (simulation, emulation, qpu)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Filter by seed(s), comma-separated (e.g., 42,123)",
    )
    parser.add_argument(
        "--study",
        type=str,
        choices=["ablation", "noise_decomposition", "sim_as_hardware"],
        default=None,
        help="Run additional study instead of main battery",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show runs without executing",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Only count runs (use with --dry-run)",
    )
    parser.add_argument(
        "--export-commands",
        action="store_true",
        help="Export one command per run for SLURM job arrays",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Log level: 0=errors, 1=progress, 2=detail (default: 1)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dict containing configuration with keys: datasets, backbones, heads, seeds,
        output_dir, ablation (optional), noise_decomposition (optional), etc.

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty: {config_path}")

    # Validate required keys
    required_keys = ["datasets", "backbones", "heads", "seeds", "output_dir"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return config


def create_run_folder(
    base_output_dir: str,
    machine_id: Optional[str],
    args: "argparse.Namespace",
    config: Dict[str, Any],
) -> str:
    """Create a numbered, descriptive folder for this execution inside base_output_dir.

    The folder name format is: NNN_<machine_id>_<description>
    where NNN auto-increments by 1 for each existing run folder.

    Args:
        base_output_dir: Root output directory (e.g. ./results)
        machine_id: Optional machine identifier
        args: Parsed CLI arguments used to build description
        config: Config dict (used to extract experiment name)

    Returns:
        Absolute path to the newly created run folder.
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # Find highest existing NNN prefix among run folders
    existing_ids = []
    for name in os.listdir(base_output_dir):
        if os.path.isdir(os.path.join(base_output_dir, name)):
            parts = name.split("_", 1)
            if parts[0].isdigit():
                existing_ids.append(int(parts[0]))
    next_id = (max(existing_ids) + 1) if existing_ids else 1

    # Build descriptive slug from CLI filters
    parts = []
    if machine_id:
        parts.append(machine_id)
    if args.dataset:
        parts.append(args.dataset.replace(",", "-"))
    if args.backbone:
        parts.append(args.backbone.replace(",", "-"))
    if args.head:
        parts.append(args.head.replace(",", "-"))
    elif args.head_type:
        parts.append(args.head_type)
    if args.study:
        parts.append(args.study)
    if not parts:
        exp_name = config.get("experiment_name", "experiment")
        parts.append(exp_name)

    slug = "_".join(parts)
    folder_name = f"{next_id:03d}_{slug}"
    run_folder = os.path.join(base_output_dir, folder_name)
    os.makedirs(run_folder, exist_ok=True)
    logger.info(f"Run folder created: {run_folder}")
    return run_folder


def get_output_paths(run_folder: str) -> Dict[str, str]:
    """Get output file paths inside a run folder.

    Args:
        run_folder: Run-specific output folder (already created)

    Returns:
        Dict with keys: runs, predictions, training_log pointing to output CSVs
    """
    return {
        "runs": os.path.join(run_folder, "runs.csv"),
        "predictions": os.path.join(run_folder, "predictions.csv"),
        "training_log": os.path.join(run_folder, "training_log.csv"),
    }


def get_head_type(head: str) -> str:
    """Classify head type based on name.

    Args:
        head: Head name (e.g., 'linear', 'pl_ideal', 'qk_noisy')

    Returns:
        Head type: 'classical', 'pennylane', or 'qiskit'
    """
    if head.startswith("pl_"):
        return "pennylane"
    elif head.startswith("qk_"):
        return "qiskit"
    else:
        return "classical"


def get_environment(head: str) -> str:
    """Classify environment (simulation, emulation, qpu) based on head.

    Args:
        head: Head name

    Returns:
        Environment: 'simulation', 'emulation', or 'qpu'
    """
    if head.endswith("_ideal") or head == "linear" or head.startswith("mlp_"):
        return "simulation"
    elif head.endswith("_noisy"):
        return "emulation"
    elif head.endswith("_real"):
        return "qpu"
    else:
        return "simulation"


def generate_main_runs(config: Dict[str, Any], args: argparse.Namespace) -> List[RunConfig]:
    """Generate runs for main battery: datasets × backbones × heads × seeds.

    Args:
        config: Configuration dictionary
        args: Parsed command-line arguments

    Returns:
        List of RunConfig objects for all combinations
    """
    runs = []

    for ds in config["datasets"]:
        ds_name = ds["name"] if isinstance(ds, dict) else ds
        for bb in config["backbones"]:
            bb_name = bb["name"] if isinstance(bb, dict) else bb
            for hd in config["heads"]:
                hd_name = hd["name"] if isinstance(hd, dict) else hd
                for seed in config["seeds"]:
                    run_id = f"{ds_name}_{bb_name}_{hd_name}_{seed}"
                    runs.append(RunConfig(
                        run_id=run_id,
                        dataset=ds_name,
                        backbone=bb_name,
                        head=hd_name,
                        seed=seed,
                        study="main",
                    ))

    return runs


def generate_ablation_runs(config: Dict[str, Any]) -> List[RunConfig]:
    """Generate runs for ablation study.

    Generates: ablation_datasets × ablation_backbones × ablation_heads × qubits × depths × ablation_seeds
    Skips the combination matching main battery defaults (n_qubits=4, depth=3).

    Args:
        config: Configuration dictionary with 'ablation' section

    Returns:
        List of RunConfig objects for ablation study
    """
    runs = []
    ablation_config = config.get("ablation", {})

    if not ablation_config:
        logger.warning("No ablation config found")
        return runs

    datasets = ablation_config.get("datasets", [])
    backbones = ablation_config.get("backbones", [])
    heads = ablation_config.get("heads", [])
    qubits_list = ablation_config.get("qubits", [4])
    depths_list = ablation_config.get("depths", [3])
    seeds = ablation_config.get("seeds", [])

    default_qubits = 4
    default_depth = 3

    for dataset in datasets:
        for backbone in backbones:
            for head in heads:
                for qubits in qubits_list:
                    for depth in depths_list:
                        # Skip default combination to avoid duplication with main battery
                        if qubits == default_qubits and depth == default_depth:
                            continue

                        for seed in seeds:
                            run_id = f"{dataset}_{backbone}_{head}_q{qubits}_d{depth}_{seed}"
                            runs.append(RunConfig(
                                run_id=run_id,
                                dataset=dataset,
                                backbone=backbone,
                                head=head,
                                seed=seed,
                                study="ablation",
                                overrides={"n_qubits": qubits, "depth": depth},
                            ))

    return runs


def generate_noise_decomposition_runs(config: Dict[str, Any]) -> List[RunConfig]:
    """Generate runs for noise decomposition study.

    For each channel in channels list:
    Generates: nd_datasets × nd_backbones × (channel_name as head) × nd_seeds

    Args:
        config: Configuration dictionary with 'noise_decomposition' section

    Returns:
        List of RunConfig objects for noise decomposition study
    """
    runs = []
    nd_config = config.get("noise_decomposition", {})

    if not nd_config:
        logger.warning("No noise_decomposition config found")
        return runs

    datasets = nd_config.get("datasets", [])
    backbones = nd_config.get("backbones", [])
    channels = nd_config.get("channels", [])
    seeds = nd_config.get("seeds", [])

    base_head = nd_config.get("base_head", "pl_noisy")

    for dataset in datasets:
        for backbone in backbones:
            for channel in channels:
                channel_name = channel.get("name", "unknown")
                for seed in seeds:
                    run_id = f"{dataset}_{backbone}_{channel_name}_{seed}"
                    runs.append(RunConfig(
                        run_id=run_id,
                        dataset=dataset,
                        backbone=backbone,
                        head=base_head,  # use base_head (pl_noisy) not channel_name
                        seed=seed,
                        study="noise_decomposition",
                        overrides={
                            "noise_channels": channel.get("noise_channels", []),
                            "n_qubits": nd_config.get("n_qubits", 4),
                            "depth": nd_config.get("depth", 3),
                        },
                    ))

    return runs


def generate_sim_as_hardware_runs(config: Dict[str, Any]) -> List[RunConfig]:
    """Generate runs for sim_as_hardware study.

    Generates: sh_datasets × sh_backbones × sh_heads × sh_seeds
    with overrides: epochs=5, shots=100

    Args:
        config: Configuration dictionary with 'sim_as_hardware' section

    Returns:
        List of RunConfig objects for sim_as_hardware study
    """
    runs = []
    sh_config = config.get("sim_as_hardware", {})

    if not sh_config:
        logger.warning("No sim_as_hardware config found")
        return runs

    datasets = sh_config.get("datasets", [])
    backbones = sh_config.get("backbones", [])
    heads = sh_config.get("heads", [])
    seeds = sh_config.get("seeds", [])

    for dataset in datasets:
        for backbone in backbones:
            for head in heads:
                for seed in seeds:
                    run_id = f"{dataset}_{backbone}_{head}_5ep_{seed}"
                    runs.append(RunConfig(
                        run_id=run_id,
                        dataset=dataset,
                        backbone=backbone,
                        head=head,
                        seed=seed,
                        study="sim_as_hardware",
                        overrides={"epochs": 5, "shots": 100},
                    ))

    return runs


def apply_filters(runs: List[RunConfig], args: argparse.Namespace) -> List[RunConfig]:
    """Apply CLI filters to reduce run set.

    Filters by: dataset, backbone, head, head_type, environment, seed

    Args:
        runs: List of RunConfig objects
        args: Parsed command-line arguments with filter values

    Returns:
        Filtered list of RunConfig objects
    """
    if args.dataset:
        allowed_datasets = set(args.dataset.split(","))
        runs = [r for r in runs if r.dataset in allowed_datasets]

    if args.backbone:
        allowed_backbones = set(args.backbone.split(","))
        runs = [r for r in runs if r.backbone in allowed_backbones]

    if args.head:
        allowed_heads = set(args.head.split(","))
        runs = [r for r in runs if r.head in allowed_heads]

    if args.head_type:
        allowed_types = set(args.head_type.split(","))
        runs = [r for r in runs if get_head_type(r.head) in allowed_types]

    if args.environment:
        allowed_envs = set(args.environment.split(","))
        runs = [r for r in runs if get_environment(r.head) in allowed_envs]

    if args.seed:
        allowed_seeds = set(int(s) for s in args.seed.split(","))
        runs = [r for r in runs if r.seed in allowed_seeds]

    return runs


def load_existing_ids(runs_csv: str) -> Set[str]:
    """Load set of existing run_ids from runs CSV for resumability.

    Args:
        runs_csv: Path to runs CSV file

    Returns:
        Set of run_id strings that already exist
    """
    if not os.path.exists(runs_csv):
        return set()

    try:
        df = pd.read_csv(runs_csv)
        if "run_id" in df.columns:
            return set(df["run_id"].astype(str))
    except Exception as e:
        logger.warning(f"Could not load existing run_ids from {runs_csv}: {e}")

    return set()


def append_to_csv(filepath: str, data: Any, is_list: bool = False) -> None:
    """Thread-safe append of data to CSV file.

    Creates file with header if it doesn't exist. Uses file locking for thread safety.

    Args:
        filepath: Path to CSV file
        data: Dict or list of dicts to append
        is_list: If True, data is expected to be a list of dicts

    Raises:
        IOError: If CSV writing fails
    """
    with _csv_lock:
        try:
            if is_list:
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])

            if os.path.exists(filepath):
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                df.to_csv(filepath, index=False)
        except Exception as e:
            logger.error(f"Failed to append to {filepath}: {e}")
            raise


def log_to_errors(errors_log: str, run_id: str, error: Exception) -> None:
    """Log error with traceback to errors.log file.

    Args:
        errors_log: Path to errors.log file
        run_id: Run identifier
        error: Exception object
    """
    with _csv_lock:
        os.makedirs(os.path.dirname(errors_log), exist_ok=True)
        with open(errors_log, "a", encoding="utf-8") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] [{run_id}] {type(error).__name__}: {str(error)}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*80 + "\n")


def log_start(machine_id: Optional[str], index: int, total: int, run_id: str) -> None:
    """Log run start message.

    Args:
        machine_id: Machine identifier
        index: Current run index (1-based)
        total: Total runs
        run_id: Run identifier
    """
    prefix = f"[{machine_id}]" if machine_id else ""
    logger.info(f"{prefix}[RUN {index}/{total}] {run_id} — started")


def log_done(
    machine_id: Optional[str],
    index: int,
    total: int,
    run_id: str,
    result: Dict[str, Any],
) -> None:
    """Log run completion with metrics.

    Args:
        machine_id: Machine identifier
        index: Current run index (1-based)
        total: Total runs
        run_id: Run identifier
        result: Result dictionary with acc, time, energy, etc.
    """
    prefix = f"[{machine_id}]" if machine_id else ""
    acc = result.get("test_accuracy", "?")
    elapsed = result.get("train_time_s", "?")
    energy = result.get("energy_kwh", None)
    energy_str = f", energy={energy:.4f}kWh" if energy else ""
    logger.info(
        f"{prefix}[DONE {index}/{total}] {run_id} — "
        f"acc={acc}, time={elapsed}s{energy_str}"
    )


def log_skip(
    machine_id: Optional[str],
    index: int,
    total: int,
    run_id: str,
) -> None:
    """Log skipped run (already exists).

    Args:
        machine_id: Machine identifier
        index: Current run index (1-based)
        total: Total runs
        run_id: Run identifier
    """
    prefix = f"[{machine_id}]" if machine_id else ""
    logger.info(f"{prefix}[SKIP {index}/{total}] {run_id} — already exists")


def log_error(
    machine_id: Optional[str],
    index: int,
    total: int,
    run_id: str,
    error: Exception,
) -> None:
    """Log error message.

    Args:
        machine_id: Machine identifier
        index: Current run index (1-based)
        total: Total runs
        run_id: Run identifier
        error: Exception object
    """
    prefix = f"[{machine_id}]" if machine_id else ""
    error_type = type(error).__name__
    logger.error(f"{prefix}[ERROR {index}/{total}] {run_id} — {error_type} (see errors.log)")


def print_dry_run_summary(runs: List[RunConfig], config: Dict[str, Any]) -> None:
    """Print dry-run summary table.

    Args:
        runs: List of runs to execute
        config: Configuration dictionary
    """
    # Categorize by environment
    simulation = [r for r in runs if get_environment(r.head) == "simulation"]
    emulation = [r for r in runs if get_environment(r.head) == "emulation"]
    qpu = [r for r in runs if get_environment(r.head) == "qpu"]

    # Group by head type for breakdown
    classical = [r for r in runs if get_head_type(r.head) == "classical"]
    pennylane = [r for r in runs if get_head_type(r.head) == "pennylane"]
    qiskit = [r for r in runs if get_head_type(r.head) == "qiskit"]

    print("\n" + "="*60)
    print("=== DRY RUN ===")
    print("="*60)
    print(f"Total runs: {len(runs)}")
    print(f"Datasets: {', '.join(sorted(set(r.dataset for r in runs)))}")
    print(f"Backbones: {', '.join(sorted(set(r.backbone for r in runs)))}")
    print(f"Heads: {', '.join(sorted(set(r.head for r in runs)))}")
    print(f"Seeds: {', '.join(str(s) for s in sorted(set(r.seed for r in runs)))}")
    print("\nBreakdown by Environment:")
    print(f"  Simulation: {len(simulation)} runs")
    print(f"  Emulation: {len(emulation)} runs")
    print(f"  QPU: {len(qpu)} runs")
    print("\nBreakdown by Framework:")
    print(f"  Classical: {len(classical)} runs")
    print(f"  PennyLane: {len(pennylane)} runs")
    print(f"  Qiskit: {len(qiskit)} runs")
    print("="*60 + "\n")


def export_commands(
    runs: List[RunConfig],
    args: argparse.Namespace,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Export one command per run for SLURM job arrays.

    Args:
        runs: List of runs to export
        args: Parsed command-line arguments (for config path, machine_id, etc.)
        config: Configuration dictionary (optional)
    """
    for run in runs:
        # Build command with all run-specific filters
        cmd = [
            "python runner.py",
            f"--config {args.config}",
        ]

        if args.machine_id:
            cmd.append(f"--machine-id {args.machine_id}")

        cmd.append(f"--dataset {run.dataset}")
        cmd.append(f"--backbone {run.backbone}")
        cmd.append(f"--head {run.head}")
        cmd.append(f"--seed {run.seed}")

        if args.study:
            cmd.append(f"--study {args.study}")

        if args.verbose != 1:
            cmd.append(f"--verbose {args.verbose}")

        print(" ".join(cmd))


def execute_run(
    run: RunConfig,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute a single experiment run.

    Imports trainer module here to avoid loading heavy dependencies (torch, pennylane, qiskit)
    during dry-run operations.

    Args:
        run: RunConfig object with experiment parameters
        config: Configuration dictionary
        args: Parsed command-line arguments

    Returns:
        Tuple of (result dict, predictions list, training_log list)

    Raises:
        ImportError: If trainer module cannot be imported
        Exception: Any exception raised during training
    """
    # Import trainer only when actually executing
    try:
        from trainer import train_and_evaluate
    except ImportError as e:
        raise ImportError(f"Could not import trainer module: {e}")

    # Execute training
    result, predictions, training_log = train_and_evaluate(
        run_config=run,
        config=config,
        overrides=run.overrides,
    )

    return result, predictions, training_log


def execute_runs_sequential(
    runs: List[RunConfig],
    config: Dict[str, Any],
    args: argparse.Namespace,
    existing_ids: Set[str],
) -> Tuple[int, int, int]:
    """Execute runs sequentially.

    Args:
        runs: List of RunConfig objects
        config: Configuration dictionary
        args: Parsed command-line arguments
        existing_ids: Set of run_ids that already exist

    Returns:
        Tuple of (completed, skipped, errors) counts
    """
    # output_dir already points to the run folder (set by main)
    run_folder = config["output_dir"]
    paths = get_output_paths(run_folder)
    errors_log = os.path.join(run_folder, "errors.log")

    completed = 0
    skipped = 0
    errors = 0
    total = len(runs)

    for index, run in enumerate(runs, 1):
        if run.run_id in existing_ids:
            log_skip(args.machine_id, index, total, run.run_id)
            skipped += 1
            continue

        log_start(args.machine_id, index, total, run.run_id)

        try:
            result, predictions, training_log = execute_run(run, config, args)
            append_to_csv(paths["runs"], result)
            append_to_csv(paths["predictions"], predictions, is_list=True)
            append_to_csv(paths["training_log"], training_log, is_list=True)
            log_done(args.machine_id, index, total, run.run_id, result)
            completed += 1
        except Exception as e:
            log_error(args.machine_id, index, total, run.run_id, e)
            log_to_errors(errors_log, run.run_id, e)
            errors += 1

    return completed, skipped, errors


def execute_runs_parallel(
    runs: List[RunConfig],
    config: Dict[str, Any],
    args: argparse.Namespace,
    existing_ids: Set[str],
) -> Tuple[int, int, int]:
    """Execute runs in parallel using ProcessPoolExecutor.

    Args:
        runs: List of RunConfig objects
        config: Configuration dictionary
        args: Parsed command-line arguments
        existing_ids: Set of run_ids that already exist

    Returns:
        Tuple of (completed, skipped, errors) counts
    """
    # output_dir already points to the run folder (set by main)
    run_folder = config["output_dir"]
    paths = get_output_paths(run_folder)
    errors_log = os.path.join(run_folder, "errors.log")

    completed = 0
    skipped = 0
    errors = 0
    total = len(runs)

    # Track which runs to execute (skip existing)
    runs_to_execute = []
    run_indices = {}

    for index, run in enumerate(runs, 1):
        if run.run_id in existing_ids:
            log_skip(args.machine_id, index, total, run.run_id)
            skipped += 1
        else:
            runs_to_execute.append(run)
            run_indices[run.run_id] = index
            log_start(args.machine_id, index, total, run.run_id)

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(execute_run, run, config, args): run
            for run in runs_to_execute
        }

        for future in as_completed(futures):
            run = futures[future]
            index = run_indices[run.run_id]

            try:
                result, predictions, training_log = future.result()
                append_to_csv(paths["runs"], result)
                append_to_csv(paths["predictions"], predictions, is_list=True)
                append_to_csv(paths["training_log"], training_log, is_list=True)
                log_done(args.machine_id, index, total, run.run_id, result)
                completed += 1
            except Exception as e:
                log_error(args.machine_id, index, total, run.run_id, e)
                log_to_errors(errors_log, run.run_id, e)
                errors += 1

    return completed, skipped, errors


def main() -> int:
    """Main entry point for experiment orchestrator.

    Returns:
        0 on success, 1 on error
    """
    args = parse_args()

    # Set logging level
    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.debug(f"Loaded config from {args.config}")

        # Generate runs based on study type
        if args.study == "ablation":
            runs = generate_ablation_runs(config)
        elif args.study == "noise_decomposition":
            runs = generate_noise_decomposition_runs(config)
        elif args.study == "sim_as_hardware":
            runs = generate_sim_as_hardware_runs(config)
        else:
            runs = generate_main_runs(config, args)

        logger.debug(f"Generated {len(runs)} runs before filtering")

        # Apply CLI filters
        runs = apply_filters(runs, args)
        logger.debug(f"After filtering: {len(runs)} runs")

        # Handle dry-run modes
        if args.dry_run:
            if args.export_commands:
                export_commands(runs, args, config)
            elif args.count:
                print(len(runs))
            else:
                print_dry_run_summary(runs, config)
            return 0

        # Get output paths and check for resumability
        # For dry-run we still create the folder so the ID is reserved
        if not args.dry_run:
            run_folder = create_run_folder(config["output_dir"], args.machine_id, args, config)
            paths = get_output_paths(run_folder)
            config["output_dir"] = run_folder
        else:
            paths = {"runs": os.path.join(config["output_dir"], "runs.csv")}
        existing_ids = load_existing_ids(paths["runs"])
        logger.info(f"Found {len(existing_ids)} existing runs, will skip them")

        # Execute runs
        if args.parallel > 1:
            logger.info(f"Starting parallel execution with {args.parallel} workers")
            completed, skipped, errors = execute_runs_parallel(runs, config, args, existing_ids)
        else:
            logger.info("Starting sequential execution")
            completed, skipped, errors = execute_runs_sequential(runs, config, args, existing_ids)

        # Print summary
        total = len(runs)
        print("\n" + "="*60)
        print("=== SUMMARY ===")
        print("="*60)
        print(f"Total runs:  {total}")
        print(f"Completed:   {completed}")
        print(f"Skipped:     {skipped}")
        print(f"Errors:      {errors}")
        print("="*60 + "\n")

        return 0 if errors == 0 else 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
bp_analysis.py - Barren Plateaus Analysis for Quantum Transfer Learning

Calculates gradient variance to detect barren plateaus in quantum circuits.
This is an independent analysis tool, not a training run.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import pennylane as qml
except ImportError:
    print("ERROR: PennyLane is required. Install with: pip install pennylane")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze barren plateaus in VQC circuits"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--machine-id",
        required=True,
        help="Machine identifier (suffix for output CSV)"
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Filter heads (comma-separated, default: all from config)"
    )
    parser.add_argument(
        "--qubits",
        default=None,
        help="Filter qubits (comma-separated, default: all from config)"
    )
    return parser.parse_args()


def build_circuit(n_qubits: int, depth: int, weights: np.ndarray, dev):
    """
    Build a parameterized quantum circuit with RY encoding and
    StronglyEntanglingLayers.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        weights: Weights for variational layers (shape: depth x n_qubits x 3)
        dev: PennyLane device

    Returns:
        QNode function
    """
    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def circuit(inputs, params):
        # RY encoding layer
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers with strong entanglement
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

        # Measurement: expectation value of Z on first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit


def add_noise_to_circuit(circuit, n_qubits: int, noise_config: Dict):
    """
    Wrap circuit with noise channels if specified in config.

    Args:
        circuit: Original circuit
        n_qubits: Number of qubits
        noise_config: Noise configuration dict

    Returns:
        Wrapped circuit with noise
    """
    if not noise_config or not noise_config.get('enabled', False):
        return circuit

    noise_channels = noise_config.get('channels', [])
    noise_params = noise_config.get('params', {})

    def noisy_circuit(inputs, params):
        result = circuit(inputs, params)
        # Note: Noise is typically applied during circuit execution in default.mixed device
        # This is a placeholder for explicit noise channel application if needed
        return result

    return noisy_circuit


def compute_gradient_variance(
    head_name: str,
    n_qubits: int,
    depth: int,
    n_initializations: int,
    use_noise: bool,
    noise_config: Dict
) -> List[Dict]:
    """
    Compute gradient variance for barren plateau detection.

    For each of n_initializations random parameter sets:
    1. Sample weights ~ U[0, 2π]
    2. Compute gradients ∂C/∂θ_i using parameter-shift rule

    Then compute variance of each gradient across all initializations.

    Args:
        head_name: Name of the VQC head (e.g., 'pl_ideal', 'pl_noisy')
        n_qubits: Number of qubits
        depth: Circuit depth
        n_initializations: Number of random initializations
        use_noise: Whether to use noise
        noise_config: Noise configuration

    Returns:
        List of result dicts with gradient statistics per parameter
    """
    logger.info(
        f"Computing gradient variance for {head_name} "
        f"({n_qubits} qubits, depth={depth})"
    )

    # Set up device
    if use_noise:
        dev = qml.device("default.mixed", wires=n_qubits)
        logger.debug("Using default.mixed device (noisy)")
    else:
        dev = qml.device("default.qubit", wires=n_qubits)
        logger.debug("Using default.qubit device (ideal)")

    # Weight shape for StronglyEntanglingLayers: (depth, n_qubits, 3)
    weight_shape = (depth, n_qubits, 3)
    n_params = int(np.prod(weight_shape))

    # Build circuit
    circuit = build_circuit(n_qubits, depth, None, dev)

    # Fixed input for gradient computation (same for all initializations)
    np.random.seed(42 + n_qubits)  # Reproducible but different per n_qubits
    fixed_input = np.random.uniform(-np.pi/2, np.pi/2, size=n_qubits)

    # Collect gradients across all initializations
    all_gradients = np.zeros((n_initializations, n_params))

    logger.info(f"Computing {n_initializations} gradient samples...")
    for init_idx in range(n_initializations):
        # Random parameter initialization
        weights = np.random.uniform(0, 2*np.pi, size=weight_shape)

        try:
            # Compute gradient w.r.t. weights using autograd
            grad_fn = qml.grad(circuit, argnum=1)
            grads = grad_fn(fixed_input, weights)
            all_gradients[init_idx] = grads.flatten()
        except Exception as e:
            logger.error(f"Error computing gradients for init {init_idx}: {e}")
            raise

        # Progress indicator
        if (init_idx + 1) % max(1, n_initializations // 10) == 0:
            print(f"  Progress: {init_idx + 1}/{n_initializations}")

    logger.info("Computing statistics per parameter...")

    # Compute statistics per parameter
    results = []
    for param_idx in range(n_params):
        grad_values = all_gradients[:, param_idx]
        results.append({
            'head': head_name,
            'n_qubits': n_qubits,
            'depth': depth,
            'param_idx': param_idx,
            'grad_mean': float(np.mean(grad_values)),
            'grad_variance': float(np.var(grad_values)),
            'grad_std': float(np.std(grad_values)),
            'n_initializations': n_initializations,
        })

    logger.info(f"Completed analysis for {head_name}: {len(results)} parameters")
    return results


def main():
    """Main entry point."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Extract barren_plateaus section
    bp_config = config.get('barren_plateaus', {})
    if not bp_config:
        logger.error("No 'barren_plateaus' section found in config")
        sys.exit(1)

    # Get parameters from config
    heads = bp_config.get('heads', ['pl_ideal', 'pl_noisy'])
    qubits_list = bp_config.get('qubits', [2, 4, 6])
    depth = bp_config.get('depth', 3)
    n_initializations = bp_config.get('n_initializations', 200)
    noise_config = bp_config.get('noise', {})

    # Filter by command-line arguments
    if args.head:
        heads = [h.strip() for h in args.head.split(',')]

    if args.qubits:
        qubits_list = [int(q.strip()) for q in args.qubits.split(',')]

    logger.info(f"Starting barren plateau analysis")
    logger.info(f"  Heads: {heads}")
    logger.info(f"  Qubits: {qubits_list}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Initializations: {n_initializations}")

    # Collect all results
    all_results = []

    # Iterate over all combinations
    n_combinations = len(heads) * len(qubits_list)
    current = 0

    for head_name in heads:
        use_noise = 'noisy' in head_name.lower()

        for n_qubits in qubits_list:
            current += 1
            print(f"\n[{current}/{n_combinations}] {head_name} with {n_qubits} qubits")

            try:
                results = compute_gradient_variance(
                    head_name=head_name,
                    n_qubits=n_qubits,
                    depth=depth,
                    n_initializations=n_initializations,
                    use_noise=use_noise,
                    noise_config=noise_config if use_noise else None
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to analyze {head_name}/{n_qubits}: {e}")
                sys.exit(1)

    # Convert to DataFrame and save
    logger.info("Saving results...")
    df = pd.DataFrame(all_results)

    output_filename = f"bp_results_{args.machine_id}.csv"
    df.to_csv(output_filename, index=False)

    print(f"\nResults saved to {output_filename}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # Summary statistics
    print(f"\nSummary:")
    for head in heads:
        head_df = df[df['head'] == head]
        print(f"  {head}: {len(head_df)} entries")


if __name__ == "__main__":
    main()

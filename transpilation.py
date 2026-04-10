#!/usr/bin/env python3
"""
transpilation.py - Circuit Transpilation Analysis for Quantum Transfer Learning

Transpiles VQC circuits to a target backend and measures:
- Original and transpiled circuit depth
- Gate counts (CX, 1-qubit gates)
- Estimated fidelity based on error rates
"""

import argparse
import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import EfficientSU2
    from qiskit.transpiler import CouplingMap
except ImportError:
    print("ERROR: Qiskit is required. Install with: pip install qiskit")
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
        description="Analyze circuit transpilation for a target backend"
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
        "--qubits",
        default=None,
        help="Filter qubits (comma-separated, default: all from config)"
    )
    return parser.parse_args()


def create_vqc_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """
    Create a VQC circuit with RY encoding and entangling layers.

    Structure:
    1. RY encoding layer (RY gates on all qubits)
    2. Repetitions of: RY + CX ladder for entanglement
    3. Final measurement-ready circuit

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth (number of entangling layers)

    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits, name=f"VQC_qubits{n_qubits}_depth{depth}")

    # RY encoding layer
    for i in range(n_qubits):
        qc.ry(np.pi/4, i)

    # Entangling layers (RY + CX ladder)
    for layer in range(depth):
        # RY rotation layer
        for i in range(n_qubits):
            qc.ry(np.pi/4, i)

        # CX entangling ladder
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        # Optional: wrap-around CX for ring connectivity
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    return qc


def count_gates(circuit: QuantumCircuit) -> Dict[str, int]:
    """
    Count gates in a circuit.

    Args:
        circuit: QuantumCircuit to analyze

    Returns:
        Dict with gate counts
    """
    gate_counts = circuit.count_ops()

    # Count CX gates
    n_cx = gate_counts.get('cx', 0)

    # Count 1-qubit gates (ry, rz, sx, x, etc.)
    n_1q = sum(count for gate, count in gate_counts.items() if gate in ['ry', 'rz', 'sx', 'x', 'id'])

    # Total gates
    n_total = sum(gate_counts.values())

    return {
        'n_cx_gates': n_cx,
        'n_1q_gates': n_1q,
        'n_total_gates': n_total,
        'gate_counts': dict(gate_counts)
    }


def estimate_fidelity(
    n_cx_gates: int,
    n_1q_gates: int,
    p_1q: float = 0.0002,
    p_2q: float = 0.005
) -> float:
    """
    Estimate circuit fidelity based on gate error rates.

    Fidelity ≈ (1 - p_2q)^n_CX × (1 - p_1q)^n_1q

    Args:
        n_cx_gates: Number of 2-qubit (CX) gates
        n_1q_gates: Number of 1-qubit gates
        p_1q: Single-qubit gate error rate
        p_2q: Two-qubit gate error rate

    Returns:
        Estimated fidelity (0 to 1)
    """
    fidelity = ((1 - p_2q) ** n_cx_gates) * ((1 - p_1q) ** n_1q_gates)
    return max(0.0, min(1.0, fidelity))  # Clamp to [0, 1]


def create_coupling_map(n_qubits: int, topology: str = "heavy_hex") -> CouplingMap:
    """
    Create a coupling map for the target backend.

    Args:
        n_qubits: Number of qubits
        topology: Topology type ('heavy_hex', 'grid', 'linear')

    Returns:
        CouplingMap
    """
    try:
        if topology == "heavy_hex":
            # Heavy hexagon topology (like IBM Torino)
            d = int(np.ceil(np.sqrt(n_qubits / 4)))
            return CouplingMap.from_heavy_hex(d)
        elif topology == "grid":
            # Grid topology
            rows = cols = int(np.ceil(np.sqrt(n_qubits)))
            couplings = []
            for r in range(rows):
                for c in range(cols):
                    i = r * cols + c
                    if i >= n_qubits:
                        break
                    # Horizontal edges
                    if c < cols - 1 and (r * cols + c + 1) < n_qubits:
                        couplings.append([i, r * cols + c + 1])
                    # Vertical edges
                    if r < rows - 1 and ((r + 1) * cols + c) < n_qubits:
                        couplings.append([i, (r + 1) * cols + c])
            return CouplingMap(couplings)
        else:  # linear
            couplings = [[i, i + 1] for i in range(n_qubits - 1)]
            return CouplingMap(couplings)
    except Exception as e:
        logger.warning(f"Could not create {topology} coupling map: {e}")
        logger.warning("Falling back to linear coupling map")
        couplings = [[i, i + 1] for i in range(n_qubits - 1)]
        return CouplingMap(couplings)


def transpile_circuit(
    circuit: QuantumCircuit,
    n_qubits: int,
    backend_name: str = "ibm_torino"
) -> QuantumCircuit:
    """
    Transpile circuit to target backend basis gates.

    Args:
        circuit: Circuit to transpile
        n_qubits: Number of qubits
        backend_name: Target backend name

    Returns:
        Transpiled QuantumCircuit
    """
    # Basis gates for IBM backends
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    # Create coupling map
    coupling_map = create_coupling_map(n_qubits, topology="heavy_hex")

    try:
        transpiled = transpile(
            circuit,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=2  # Moderate optimization
        )
        return transpiled
    except Exception as e:
        logger.error(f"Transpilation failed: {e}")
        raise


def analyze_circuit(
    n_qubits: int,
    depth: int,
    backend_name: str = "ibm_torino"
) -> Dict:
    """
    Analyze a VQC circuit: original depth, transpiled depth, and fidelity.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        backend_name: Target backend name

    Returns:
        Dict with analysis results
    """
    logger.info(f"Analyzing circuit: {n_qubits} qubits, depth {depth}")

    # Create original circuit
    circuit = create_vqc_circuit(n_qubits, depth)
    logger.debug(f"Original circuit: {circuit.num_qubits} qubits, {circuit.num_clbits} classical bits")

    # Get original depth and gate counts
    depth_original = circuit.depth()
    gates_original = count_gates(circuit)

    logger.info(f"  Original circuit depth: {depth_original}")
    logger.info(f"  Original gates: CX={gates_original['n_cx_gates']}, 1Q={gates_original['n_1q_gates']}")

    # Transpile circuit
    try:
        transpiled = transpile_circuit(circuit, n_qubits, backend_name)
    except Exception as e:
        logger.error(f"Transpilation failed for {n_qubits} qubits: {e}")
        raise

    # Get transpiled depth and gate counts
    depth_transpiled = transpiled.depth()
    gates_transpiled = count_gates(transpiled)

    logger.info(f"  Transpiled circuit depth: {depth_transpiled}")
    logger.info(f"  Transpiled gates: CX={gates_transpiled['n_cx_gates']}, 1Q={gates_transpiled['n_1q_gates']}")

    # Estimate fidelity based on transpiled circuit
    estimated_fidelity = estimate_fidelity(
        gates_transpiled['n_cx_gates'],
        gates_transpiled['n_1q_gates']
    )

    logger.info(f"  Estimated fidelity: {estimated_fidelity:.6f}")

    return {
        'n_qubits': n_qubits,
        'depth_original': depth_original,
        'depth_transpiled': depth_transpiled,
        'depth_ratio': depth_transpiled / depth_original if depth_original > 0 else 1.0,
        'n_cx_gates': gates_transpiled['n_cx_gates'],
        'n_1q_gates': gates_transpiled['n_1q_gates'],
        'n_total_gates': gates_transpiled['n_total_gates'],
        'estimated_fidelity': estimated_fidelity,
        'backend': backend_name,
    }


def main():
    """Main entry point."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Extract transpilation section
    tp_config = config.get('transpilation', {})
    if not tp_config:
        logger.error("No 'transpilation' section found in config")
        sys.exit(1)

    # Get parameters from config
    qubits_list = tp_config.get('qubits', [2, 4, 6, 8])
    depth = tp_config.get('depth', 3)
    backend_name = tp_config.get('backend', 'ibm_torino')

    # Filter by command-line arguments
    if args.qubits:
        qubits_list = [int(q.strip()) for q in args.qubits.split(',')]

    logger.info(f"Starting circuit transpilation analysis")
    logger.info(f"  Qubits: {qubits_list}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Backend: {backend_name}")

    # Collect all results
    all_results = []

    # Analyze each circuit
    for i, n_qubits in enumerate(qubits_list):
        print(f"\n[{i+1}/{len(qubits_list)}] Analyzing {n_qubits} qubits...")

        try:
            result = analyze_circuit(n_qubits, depth, backend_name)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {n_qubits} qubits: {e}")
            # Continue with other circuits
            continue

    if not all_results:
        logger.error("No circuits analyzed successfully")
        sys.exit(1)

    # Convert to DataFrame and save
    logger.info("Saving results...")
    df = pd.DataFrame(all_results)

    output_filename = f"transpilation_results_{args.machine_id}.csv"
    df.to_csv(output_filename, index=False)

    print(f"\nResults saved to {output_filename}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # Summary statistics
    print(f"\nSummary:")
    print(f"  Min fidelity: {df['estimated_fidelity'].min():.6f}")
    print(f"  Max fidelity: {df['estimated_fidelity'].max():.6f}")
    print(f"  Mean depth ratio (transpiled/original): {df['depth_ratio'].mean():.2f}")


if __name__ == "__main__":
    main()

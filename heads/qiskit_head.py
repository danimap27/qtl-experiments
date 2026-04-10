"""
Qiskit VQC head: Variational quantum circuit using Qiskit framework.

Implements a quantum circuit using Qiskit with support for ideal and noisy simulations.
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.primitives import Estimator, Sampler
    from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
except ImportError:
    raise ImportError(
        "Qiskit is required for QiskitHead. Install with: "
        "pip install qiskit qiskit-machine-learning qiskit-aer"
    )

logger = logging.getLogger(__name__)


class QiskitHead(nn.Module):
    """
    Variational Quantum Circuit (VQC) classification head using Qiskit.

    Architecture:
    1. Classical projection: Linear(feature_dim -> n_qubits, bias=False)
    2. Angle encoding: tanh(x) * pi/2
    3. Quantum circuit:
       - RY encoding layer
       - Entangling layers with RX, RY, RX rotations and CNOT chain
       - Optional depolarizing noise
    4. Measurement: Pauli-Z expectation values (Estimator) or bitstring sampling (Sampler)
    5. Classical output: Linear(n_qubits -> num_classes)

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        n_qubits: Number of qubits (default: 4)
        depth: Circuit depth / number of entangling layers (default: 3)
        noise: Enable noise simulation (default: False)
        noise_channels: List of noise channel types to apply (default: None)
        noise_params: Noise parameter configuration (default: None)
        shots: Number of measurement shots for Sampler (default: 1024)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        n_qubits: int = 4,
        depth: int = 3,
        noise: bool = False,
        noise_channels: Optional[List[str]] = None,
        noise_params: Optional[Dict[str, Any]] = None,
        shots: int = 1024,
    ):
        """Initialize the Qiskit VQC head."""
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.depth = depth
        self.noise = noise
        self.noise_channels = noise_channels or []
        self.noise_params = noise_params or {}
        self.shots = shots

        # Classical projection layer
        self.proj = nn.Linear(feature_dim, n_qubits, bias=False)

        # Variational parameters: (depth, n_qubits, 3)
        # Each of depth layers has 3 rotation angles per qubit
        self.num_weights = depth * n_qubits * 3
        self.register_buffer("_weight_shape", torch.tensor([depth, n_qubits, 3]))

        # Register actual trainable weight parameter
        self.q_weights = nn.Parameter(0.01 * torch.randn(self.num_weights))

        # Classical output layer
        self.output = nn.Linear(n_qubits, num_classes)

        # Create Qiskit QNN and wrap with TorchConnector
        self._create_qnn()

        logger.info(
            f"QiskitHead initialized: {n_qubits} qubits, depth={depth}, "
            f"noise={noise}, shots={shots}"
        )

    def _create_qnn(self):
        """Create the Qiskit quantum neural network."""
        # Create a parametrized circuit
        def create_circuit(inputs, weights):
            """
            Create a parametrized quantum circuit.

            Args:
                inputs: Feature inputs (n_qubits,) numpy array
                weights: Variational weights (depth * n_qubits * 3,) numpy array

            Returns:
                QuantumCircuit with measurements
            """
            qc = QuantumCircuit(self.n_qubits)

            # Reshape weights back to (depth, n_qubits, 3)
            weights_reshaped = weights.reshape(self.depth, self.n_qubits, 3)

            # RY encoding layer
            for i in range(self.n_qubits):
                qc.ry(inputs[i], i)

            # Entangling layers
            for layer_idx in range(self.depth):
                # Single-qubit rotations: RX, RY, RX
                for i in range(self.n_qubits):
                    qc.rx(weights_reshaped[layer_idx, i, 0], i)
                    qc.ry(weights_reshaped[layer_idx, i, 1], i)
                    qc.rx(weights_reshaped[layer_idx, i, 2], i)

                # Entangling: CNOT chain
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                # Wrap-around CNOT
                qc.cx(self.n_qubits - 1, 0)

            return qc

        # Use Estimator for ideal case (no shots, exact expectation values)
        if not self.noise:
            # Create QNN that measures Pauli-Z on each qubit
            observables = [
                f"Z" + "I" * (self.n_qubits - 1 - i) + "I" * i
                for i in range(self.n_qubits)
            ]

            def circuit_factory(weights):
                """Create circuits for all qubits."""
                # This returns a list of circuits, one per observable
                return [create_circuit] * len(observables)

            # Use EstimatorQNN for expectation value computation
            estimator = Estimator()
            self.qnn = EstimatorQNN(
                circuit=create_circuit,
                estimator=estimator,
                input_params=[f"x{i}" for i in range(self.n_qubits)],
                weight_params=[f"w{i}" for i in range(self.num_weights)],
                observables=observables,
            )

        else:
            # Use AerSimulator with noise for noisy case
            simulator = AerSimulator()
            noise_model = self._build_noise_model()
            simulator.set_options(noise_model=noise_model)

            # Use SamplerQNN for shot-based measurement
            sampler = Sampler(backend=simulator, options={"shots": self.shots})

            # Interpretation function: convert bitstrings to measurement outcomes
            def interpret_bitstring(bitstring, shots):
                """Interpret measurement bitstring as Pauli-Z expectation values."""
                # Convert bitstring to list of 0/1 outcomes
                # For Pauli-Z: <Z> = P(0) - P(1)
                outcomes = []
                for i in range(self.n_qubits):
                    count_0 = sum(1 for b in bitstring if b[i] == "0")
                    count_1 = shots - count_0
                    expectation = (count_0 - count_1) / shots
                    outcomes.append(expectation)
                return outcomes

            # Add classical bits for measurement
            def circuit_with_measurement(inputs, weights):
                qc = create_circuit(inputs, weights)
                qc.measure_all()
                return qc

            self.qnn = SamplerQNN(
                circuit=circuit_with_measurement,
                sampler=sampler,
                input_params=[f"x{i}" for i in range(self.n_qubits)],
                weight_params=[f"w{i}" for i in range(self.num_weights)],
                interpret=interpret_bitstring,
            )

        # Wrap QNN with TorchConnector for PyTorch integration
        self.qnn_torch = TorchConnector(self.qnn)

    def _build_noise_model(self) -> NoiseModel:
        """
        Build a noise model from IBM Heron r2 calibration parameters.

        Returns:
            Qiskit NoiseModel
        """
        noise_model = NoiseModel()

        # Get noise parameters
        p1q = self.noise_params.get("p1q", 0.001)
        p2q = self.noise_params.get("p2q", 0.005)
        readout_error = self.noise_params.get("readout_error", 0.01)

        if "depolarizing" in self.noise_channels:
            # Single-qubit gate errors
            error_1q = depolarizing_error(p1q, 1)
            noise_model.add_all_qubit_quantum_error(
                error_1q, ["rx", "ry", "rz", "x", "y", "z", "h", "s", "t"]
            )

            # Two-qubit gate errors
            error_2q = depolarizing_error(p2q, 2)
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cy", "cz", "swap"])

        if "readout" in self.noise_channels:
            # Readout errors
            ro_error = ReadoutError(
                [
                    [1 - readout_error, readout_error],
                    [readout_error, 1 - readout_error],
                ]
            )
            noise_model.add_all_qubit_readout_error(ro_error)

        return noise_model

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Classical projection and angle encoding
        x_proj = torch.tanh(self.proj(x)) * (np.pi / 2)

        # Use the registered weight parameter
        weights_flat = self.q_weights

        # Run QNN on batch
        # The QNN expects (batch_size, n_qubits + num_weights)
        # Input: x_proj (batch_size, n_qubits)
        # Weights: weights_flat (num_weights,) - broadcast to batch
        q_outputs = []
        for i in range(batch_size):
            q_input = x_proj[i].unsqueeze(0)  # (1, n_qubits)
            q_out = self.qnn_torch(q_input, weights_flat)  # (1, n_qubits)
            q_outputs.append(q_out)

        # Stack results: (batch_size, n_qubits)
        q_out_tensor = torch.cat(q_outputs, dim=0)

        # Classical output layer
        return self.output(q_out_tensor)


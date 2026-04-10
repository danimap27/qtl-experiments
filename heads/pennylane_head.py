"""
PennyLane VQC head: Variational quantum circuit classification head.

Implements a quantum circuit using PennyLane with support for ideal and noisy simulations.
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    import pennylane as qml
except ImportError:
    raise ImportError("PennyLane is required for PennyLaneHead. Install with: pip install pennylane")

logger = logging.getLogger(__name__)


class PennyLaneHead(nn.Module):
    """
    Variational Quantum Circuit (VQC) classification head using PennyLane.

    Architecture:
    1. Classical projection: Linear(feature_dim -> n_qubits, bias=False)
    2. Angle encoding: tanh(x) * pi/2
    3. Quantum circuit:
       - RY encoding layer
       - StronglyEntanglingLayers with depth layers
       - Optional noise channels
    4. Measurement: Pauli-Z expectation values on each qubit
    5. Classical output: Linear(n_qubits -> num_classes)

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        n_qubits: Number of qubits (default: 4)
        depth: Circuit depth / number of entangling layers (default: 3)
        backend: PennyLane device backend (default: 'default.qubit')
        noise: Enable noise simulation (default: False)
        noise_channels: List of noise channel types to apply (default: None)
        noise_params: Noise parameter configuration (default: None)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        n_qubits: int = 4,
        depth: int = 3,
        backend: str = "default.qubit",
        noise: bool = False,
        noise_channels: Optional[List[str]] = None,
        noise_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the PennyLane VQC head."""
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.noise = noise
        self.noise_channels = noise_channels or []
        self.noise_params = noise_params or {}

        # Classical projection layer
        self.proj = nn.Linear(feature_dim, n_qubits, bias=False)

        # Variational parameters for StronglyEntanglingLayers
        # Shape: (depth, n_qubits, 3) for 3 rotation angles per qubit per layer
        self.register_parameter(
            "weights",
            nn.Parameter(0.01 * torch.randn(depth, n_qubits, 3)),
        )

        # Classical output layer
        self.output = nn.Linear(n_qubits, num_classes)

        # Create quantum device and qnode
        self._create_qnode()

        logger.info(
            f"PennyLaneHead initialized: {n_qubits} qubits, depth={depth}, "
            f"backend={backend}, noise={noise}"
        )

    def _create_qnode(self):
        """Create the PennyLane quantum node (circuit)."""
        # Select device with or without noise
        if self.noise and self.backend == "default.qubit":
            # Use mixed state simulator for noisy simulations
            device_name = "default.mixed"
        else:
            device_name = self.backend

        dev = qml.device(device_name, wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            """
            Quantum circuit with RY encoding and entangling layers.

            Args:
                inputs: Encoded input angles (n_qubits,)
                weights: Variational parameters (depth, n_qubits, 3)

            Returns:
                List of Pauli-Z expectation values for each qubit
            """
            # RY encoding layer
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational entangling layers (StronglyEntanglingLayers)
            if self.noise and self.noise_channels:
                # Manual construction for noise insertion
                self._apply_entangling_layers_with_noise(weights)
            else:
                # Use built-in template for efficiency
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Measurement: expectation values of Pauli-Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def _apply_entangling_layers_with_noise(self, weights):
        """
        Apply entangling layers with noise channels inserted.

        Uses the same rotation pattern as StronglyEntanglingLayers but allows
        noise insertion between gates. Supports all 3 isolated channels for
        the noise decomposition study:
        - amplitude_damping: gamma = 1 - exp(-t_gate / T1)
        - phase_damping: gamma_phi (pure dephasing contribution)
        - depolarizing: p = p1q (1-qubit), p2q (2-qubit)

        Args:
            weights: Variational parameters (depth, n_qubits, 3)
        """
        # Get noise parameters (IBM Heron r2 calibration)
        p1q = self.noise_params.get("p1q", 0.0002)
        p2q = self.noise_params.get("p2q", 0.005)
        T1_us = self.noise_params.get("T1_us", 250)
        T2_us = self.noise_params.get("T2_us", 150)
        t1q_ns = self.noise_params.get("t1q_ns", 32)
        t2q_ns = self.noise_params.get("t2q_ns", 68)

        # Compute damping probabilities from calibration data
        import math
        T1_ns = T1_us * 1000
        T2_ns = T2_us * 1000
        gamma_1q = 1 - math.exp(-t1q_ns / T1_ns) if T1_ns > 0 else 0
        gamma_2q = 1 - math.exp(-t2q_ns / T1_ns) if T1_ns > 0 else 0
        # Pure dephasing: 1/T_phi = 1/T2 - 1/(2*T1)
        T_phi_inv = (1.0 / T2_ns) - (1.0 / (2 * T1_ns)) if T2_ns > 0 else 0
        gamma_phi_1q = 1 - math.exp(-t1q_ns * T_phi_inv) if T_phi_inv > 0 else 0
        gamma_phi_2q = 1 - math.exp(-t2q_ns * T_phi_inv) if T_phi_inv > 0 else 0

        def _apply_noise_1q(wire):
            """Apply active noise channels after a 1-qubit gate."""
            if "amplitude_damping" in self.noise_channels:
                qml.AmplitudeDamping(gamma_1q, wires=wire)
            if "phase_damping" in self.noise_channels:
                qml.PhaseDamping(gamma_phi_1q, wires=wire)
            if "depolarizing" in self.noise_channels:
                qml.DepolarizingChannel(p1q, wires=wire)

        def _apply_noise_2q(wire0, wire1):
            """Apply active noise channels after a 2-qubit gate."""
            if "amplitude_damping" in self.noise_channels:
                qml.AmplitudeDamping(gamma_2q, wires=wire0)
                qml.AmplitudeDamping(gamma_2q, wires=wire1)
            if "phase_damping" in self.noise_channels:
                qml.PhaseDamping(gamma_phi_2q, wires=wire0)
                qml.PhaseDamping(gamma_phi_2q, wires=wire1)
            if "depolarizing" in self.noise_channels:
                # Apply 1-qubit depolarizing to each qubit (approximation of 2q error)
                qml.DepolarizingChannel(p2q, wires=wire0)
                qml.DepolarizingChannel(p2q, wires=wire1)

        for layer_idx in range(self.depth):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RX(weights[layer_idx, i, 2], wires=i)
                _apply_noise_1q(i)

            # Entangling layer (CNOT chain)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                _apply_noise_2q(i, i + 1)

            # Wrap-around CNOT
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                _apply_noise_2q(self.n_qubits - 1, 0)

    def forward(self, x):
        """
        Forward pass with batch processing.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Classical projection and angle encoding
        x_proj = torch.tanh(self.proj(x)) * (np.pi / 2)

        # Run quantum circuit on each sample
        # PennyLane processes one sample at a time
        q_outputs = []
        for i in range(batch_size):
            # Circuit expects (n_qubits,) input and returns list of expvals
            q_out = self.circuit(x_proj[i], self.weights)
            q_outputs.append(q_out)

        # Stack results: (batch_size, n_qubits)
        q_out_tensor = torch.stack(q_outputs)

        # Classical output layer
        return self.output(q_out_tensor)

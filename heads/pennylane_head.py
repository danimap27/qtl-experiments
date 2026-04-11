"""
PennyLane VQC head: Variational quantum circuit classification head.

Implements a quantum circuit using PennyLane with support for ideal and noisy
simulations. Mirrors the QiskitHead design for a fair framework comparison.

Performance / device selection (symmetric with QiskitHead):
  gradient_method="adjoint"     : Adjoint (reverse-mode) differentiation via
                                   lightning.qubit or lightning.gpu. O(depth)
                                   circuit evaluations — equivalent to Qiskit's
                                   ReverseEstimatorGradient. Ideal circuits only.
  gradient_method="backprop"    : Full autograd through the statevector (default.qubit)
                                   or density matrix (default.mixed for noisy).
                                   Fast for shallow circuits; memory scales as 2^n.
  gradient_method="param_shift" : Exact parameter-shift. O(2*n_params) evals.
                                   Works on all devices including noisy mixed-state.

  device_preference="auto"  : lightning.gpu (if CUDA+cuQuantum) → lightning.qubit
                               → default.qubit. For noisy, always default.mixed.
  device_preference="gpu"   : Force lightning.gpu; error if unavailable.
  device_preference="cpu"   : lightning.qubit (adjoint) or default.qubit (backprop).
"""

import logging
import math
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    import pennylane as qml
except ImportError:
    raise ImportError("PennyLane is required for PennyLaneHead. Install with: pip install pennylane")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device / diff_method compatibility matrix
# ---------------------------------------------------------------------------
# device          | adjoint | backprop | param_shift | noise
# lightning.gpu   |   ✓     |    ✗     |     ✓       |   ✗
# lightning.qubit |   ✓     |    ✗     |     ✓       |   ✗
# default.qubit   |   ✗     |    ✓     |     ✓       |   ✗
# default.mixed   |   ✗     |    ✓     |     ✓       |   ✓
# ---------------------------------------------------------------------------


class PennyLaneHead(nn.Module):
    """
    Variational Quantum Circuit (VQC) classification head using PennyLane.

    Architecture:
    1. Classical projection: Linear(feature_dim -> n_qubits, bias=False)
    2. Angle encoding: tanh(x) * pi/2
    3. Quantum circuit:
       - RY encoding layer
       - StronglyEntanglingLayers with depth layers
       - Optional IBM Heron r2-calibrated noise channels
    4. Measurement: Pauli-Z expectation values on each qubit
    5. Classical output: Linear(n_qubits -> num_classes)

    Args:
        feature_dim:        Input feature dimension.
        num_classes:        Number of output classes.
        n_qubits:           Number of qubits (default: 4).
        depth:              Number of entangling layers (default: 3).
        backend:            Explicit PennyLane device name. If set, overrides
                            device_preference auto-selection.
        noise:              Enable noise simulation (default: False).
        noise_channels:     Active noise channels: 'amplitude_damping',
                            'phase_damping', 'depolarizing'.
        noise_params:       IBM Heron r2 calibration values dict.
        gradient_method:    'adjoint' | 'backprop' | 'param_shift'.
                            Defaults to 'adjoint' (ideal) or 'backprop' (noisy).
        device_preference:  'auto' | 'cpu' | 'gpu'. Ignored when backend is set.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        n_qubits: int = 4,
        depth: int = 3,
        backend: str = "",
        noise: bool = False,
        noise_channels: Optional[List[str]] = None,
        noise_params: Optional[Dict[str, Any]] = None,
        gradient_method: str = "adjoint",
        device_preference: str = "auto",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.noise = noise
        self.noise_channels = noise_channels or []
        self.noise_params = noise_params or {}
        self.device_preference = device_preference

        # Noisy circuits require default.mixed which only supports backprop/param_shift
        if noise and gradient_method == "adjoint":
            logger.warning(
                "gradient_method='adjoint' is not supported with noisy simulation "
                "(requires lightning device). Switching to 'backprop'."
            )
            gradient_method = "backprop"
        self.gradient_method = gradient_method

        # Classical projection layer (no bias — keeps trainable param count low)
        self.proj = nn.Linear(feature_dim, n_qubits, bias=False)

        # Variational parameters: (depth, n_qubits, 3) rotations per layer
        self.register_parameter(
            "weights",
            nn.Parameter(0.01 * torch.randn(depth, n_qubits, 3)),
        )

        # Classical output layer
        self.output = nn.Linear(n_qubits, num_classes)

        # Create quantum device and qnode
        self._actual_device, self._actual_diff_method = self._create_qnode()

        logger.info(
            f"PennyLaneHead initialized: {n_qubits} qubits, depth={depth}, "
            f"device={self._actual_device}, diff_method={self._actual_diff_method}, "
            f"noise={noise}"
        )

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    def _select_device(self) -> qml.Device:
        """
        Choose the best available PennyLane device.

        Priority (ideal):  lightning.gpu  →  lightning.qubit  →  default.qubit
        Priority (noisy):  default.mixed  (always — only device supporting channels)

        Returns (device, resolved_device_name).
        """
        if self.noise:
            # Mixed-state simulation required for noise channels
            dev = qml.device("default.mixed", wires=self.n_qubits)
            return dev, "default.mixed"

        # Explicit backend override
        if self.backend:
            dev = qml.device(self.backend, wires=self.n_qubits)
            return dev, self.backend

        candidates = []
        if self.device_preference in ("auto", "gpu"):
            candidates.append("lightning.gpu")
        if self.device_preference in ("auto", "cpu"):
            candidates += ["lightning.qubit", "default.qubit"]
        elif self.device_preference == "gpu":
            candidates.append("lightning.qubit")   # fallback if GPU unavailable

        for name in candidates:
            try:
                dev = qml.device(name, wires=self.n_qubits)
                return dev, name
            except Exception as exc:
                logger.debug(f"Device '{name}' unavailable: {exc}")

        # Final fallback
        dev = qml.device("default.qubit", wires=self.n_qubits)
        return dev, "default.qubit"

    def _resolve_diff_method(self, device_name: str) -> str:
        """
        Resolve the effective diff_method given device constraints.

          adjoint   → only with lightning.* devices
          backprop  → only with default.qubit / default.mixed
          param_shift → all devices
        """
        requested = self.gradient_method
        is_lightning = device_name.startswith("lightning")
        is_mixed = device_name == "default.mixed"

        if requested == "adjoint":
            if not is_lightning:
                logger.warning(
                    f"'adjoint' requires lightning.* (got '{device_name}'). "
                    f"Falling back to 'backprop'."
                )
                return "backprop"
            return "adjoint"

        if requested == "backprop":
            if is_lightning:
                logger.warning(
                    f"'backprop' is not supported on '{device_name}'. "
                    f"Falling back to 'adjoint'."
                )
                return "adjoint"
            return "backprop"

        return "param_shift"  # always valid

    # ------------------------------------------------------------------
    # QNode construction
    # ------------------------------------------------------------------

    def _create_qnode(self):
        """
        Build the PennyLane quantum node.

        Returns (device_name, diff_method) actually used, for logging.
        """
        dev, device_name = self._select_device()
        diff_method = self._resolve_diff_method(device_name)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            """
            Quantum circuit: RY encoding + StronglyEntanglingLayers + PauliZ measurements.

            Args:
                inputs:  Encoded input angles, shape (n_qubits,)
                weights: Variational parameters, shape (depth, n_qubits, 3)

            Returns:
                List of Pauli-Z expectation values [E[Z_0], ..., E[Z_{n-1}]]
            """
            # Angle encoding layer
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational entangling layers
            if self.noise and self.noise_channels:
                self._apply_entangling_layers_with_noise(weights)
            else:
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        return device_name, diff_method

    # ------------------------------------------------------------------
    # Noise layer construction (IBM Heron r2 calibration)
    # ------------------------------------------------------------------

    def _apply_entangling_layers_with_noise(self, weights):
        """
        Apply entangling layers with IBM Heron r2-calibrated noise channels.

        Supports isolated channel study (6 channels):
          - amplitude_damping : T1 relaxation (gamma = 1 - exp(-t/T1))
          - phase_damping      : pure dephasing (gamma_phi from T2, T1)
          - depolarizing       : gate error rates (p1q, p2q)

        Args:
            weights: Variational parameters (depth, n_qubits, 3)
        """
        p1q     = self.noise_params.get("p1q", 0.0002)
        p2q     = self.noise_params.get("p2q", 0.005)
        T1_us   = self.noise_params.get("T1_us", 250)
        T2_us   = self.noise_params.get("T2_us", 150)
        t1q_ns  = self.noise_params.get("t1q_ns", 32)
        t2q_ns  = self.noise_params.get("t2q_ns", 68)

        T1_ns = T1_us * 1000
        T2_ns = T2_us * 1000

        gamma_1q  = 1 - math.exp(-t1q_ns / T1_ns)  if T1_ns > 0 else 0
        gamma_2q  = 1 - math.exp(-t2q_ns / T1_ns)  if T1_ns > 0 else 0
        T_phi_inv = (1.0 / T2_ns - 1.0 / (2 * T1_ns)) if T2_ns > 0 else 0
        gphi_1q   = 1 - math.exp(-t1q_ns * T_phi_inv) if T_phi_inv > 0 else 0
        gphi_2q   = 1 - math.exp(-t2q_ns * T_phi_inv) if T_phi_inv > 0 else 0

        def _noise_1q(wire):
            if "amplitude_damping" in self.noise_channels:
                qml.AmplitudeDamping(gamma_1q, wires=wire)
            if "phase_damping" in self.noise_channels:
                qml.PhaseDamping(gphi_1q, wires=wire)
            if "depolarizing" in self.noise_channels:
                qml.DepolarizingChannel(p1q, wires=wire)

        def _noise_2q(w0, w1):
            if "amplitude_damping" in self.noise_channels:
                qml.AmplitudeDamping(gamma_2q, wires=w0)
                qml.AmplitudeDamping(gamma_2q, wires=w1)
            if "phase_damping" in self.noise_channels:
                qml.PhaseDamping(gphi_2q, wires=w0)
                qml.PhaseDamping(gphi_2q, wires=w1)
            if "depolarizing" in self.noise_channels:
                qml.DepolarizingChannel(p2q, wires=w0)
                qml.DepolarizingChannel(p2q, wires=w1)

        for layer in range(self.depth):
            for i in range(self.n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RX(weights[layer, i, 2], wires=i)
                _noise_1q(i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                _noise_2q(i, i + 1)
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                _noise_2q(self.n_qubits - 1, 0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical projection → quantum circuit → classical output.

        Args:
            x: Input features, shape (batch_size, feature_dim)

        Returns:
            Logits, shape (batch_size, num_classes)
        """
        # Classical projection + angle encoding → [-pi/2, pi/2]
        x_proj = torch.tanh(self.proj(x)) * (np.pi / 2)   # (batch_size, n_qubits)

        # Run circuit sample-by-sample.
        #
        # Note on batching strategy:
        #   - With diff_method="adjoint" (lightning.*): a single reverse-mode sweep
        #     computes ALL parameter gradients in O(depth) evals regardless of
        #     batch size. The per-sample loop adds only Python overhead (~negligible
        #     vs. the O(2^n) statevector ops).
        #   - With diff_method="backprop" (default.*): PyTorch autograd traces
        #     through the circuit. torch.vmap would batch this, but compatibility
        #     with QNode closures is device-dependent; the loop is used for safety.
        #   - Symmetric with QiskitHead forward which batches via TorchConnector.
        q_outputs = []
        for i in range(x_proj.shape[0]):
            q_out = self.circuit(x_proj[i], self.weights)
            # QNode returns a list of 0-d tensors (one per measured qubit).
            # Stack into 1-D tensor (n_qubits,) before accumulating the batch.
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack([
                    o if isinstance(o, torch.Tensor)
                    else torch.as_tensor(o, dtype=torch.float32)
                    for o in q_out
                ])
            elif not isinstance(q_out, torch.Tensor):
                q_out = torch.as_tensor(q_out, dtype=torch.float32)
            q_outputs.append(q_out)

        q_out_tensor = torch.stack(q_outputs)   # (batch_size, n_qubits)
        return self.output(q_out_tensor)         # (batch_size, num_classes)

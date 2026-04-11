"""
Qiskit VQC head: Variational quantum circuit using Qiskit framework.

Updated for Qiskit 1.x API (V2 primitives: StatevectorEstimator, AerEstimatorV2).
Uses ParameterVector for proper parametrized circuits and SparsePauliOp for observables.

Performance notes:
- gradient_method="reverse"  : Adjoint differentiation — O(depth) circuit evals, ~10-30x faster
                               than parameter-shift for StatevectorEstimator (ideal only).
- gradient_method="spsa"     : Stochastic approximation — O(1) evals per step, fastest but
                               introduces gradient noise; suitable for large param counts.
- gradient_method="param_shift": Default parameter-shift, exact but O(2*n_params) evals.
- forward() always batches the full (batch_size, n_qubits) input in a single QNN call
  to eliminate per-sample Python loop overhead.
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
except ImportError as _e:
    raise ImportError(
        f"Qiskit is required for QiskitHead. Install with: "
        f"pip install qiskit qiskit-machine-learning qiskit-aer  (error: {_e})"
    )

logger = logging.getLogger(__name__)


class QiskitHead(nn.Module):
    """
    Variational Quantum Circuit (VQC) classification head using Qiskit.

    Architecture:
    1. Classical projection: Linear(feature_dim -> n_qubits, bias=False)
    2. Angle encoding: tanh(x) * pi/2
    3. Quantum circuit:
       - RY encoding layer (input_params)
       - Entangling layers: RX/RY/RX rotations + CNOT chain (weight_params)
       - Optional depolarizing + readout noise (IBM Heron r2 calibrated)
    4. Measurement: PauliZ expectation values via EstimatorQNN
    5. Classical output: Linear(n_qubits -> num_classes)

    Weights are managed by TorchConnector (registered as nn.Parameter automatically).

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        n_qubits: Number of qubits (default: 4)
        depth: Circuit depth / number of entangling layers (default: 3)
        noise: Enable noise simulation (default: False)
        noise_channels: List of noise channel types to apply (default: None)
        noise_params: Noise parameter configuration (default: None)
        shots: Number of measurement shots for noisy simulation (default: 1024)
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
        gradient_method: str = "reverse",
        device_preference: str = "auto",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.depth = depth
        self.noise = noise
        self.noise_channels = noise_channels or []
        self.noise_params = noise_params or {}
        self.shots = shots
        self.device_preference = device_preference
        # reverse/adjoint only works with StatevectorEstimator (ideal, CPU).
        # For noisy (AerEstimatorV2) fall back to param_shift.
        self.gradient_method = gradient_method if not noise else "param_shift"

        # Number of variational parameters: depth * n_qubits * 3 (RX, RY, RX per qubit per layer)
        self.num_weights = depth * n_qubits * 3

        # Classical projection: feature_dim -> n_qubits (no bias, keeps param count low)
        self.proj = nn.Linear(feature_dim, n_qubits, bias=False)

        # Classical output layer
        self.output = nn.Linear(n_qubits, num_classes)

        # Build QNN + TorchConnector (registers quantum weights as nn.Parameter)
        self._create_qnn()

        logger.info(
            f"QiskitHead initialized: {n_qubits} qubits, depth={depth}, "
            f"noise={noise}, channels={self.noise_channels}, shots={shots}, "
            f"gradient={self.gradient_method}, device_preference={device_preference}"
        )

    def _build_circuit(self):
        """
        Build a parametrized quantum circuit with ParameterVector.

        Returns:
            (QuantumCircuit, input_params ParameterVector, weight_params ParameterVector)
        """
        input_params = ParameterVector("x", self.n_qubits)
        weight_params = ParameterVector("w", self.num_weights)

        qc = QuantumCircuit(self.n_qubits)

        # RY encoding layer
        for i in range(self.n_qubits):
            qc.ry(input_params[i], i)

        # Entangling layers: RX/RY/RX + CNOT chain
        w_idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                qc.rx(weight_params[w_idx], i);     w_idx += 1
                qc.ry(weight_params[w_idx], i);     w_idx += 1
                qc.rx(weight_params[w_idx], i);     w_idx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            # Wrap-around CNOT for full entanglement
            if self.n_qubits > 1:
                qc.cx(self.n_qubits - 1, 0)

        return qc, input_params, weight_params

    def _build_observables(self) -> List[SparsePauliOp]:
        """
        Build PauliZ observables, one per qubit.

        Qiskit Pauli strings are right-to-left (qubit 0 = rightmost character).
        Observable for qubit i: I^(n-1-i) Z I^i

        Returns:
            List of SparsePauliOp, length = n_qubits
        """
        observables = []
        for i in range(self.n_qubits):
            pauli_str = "I" * (self.n_qubits - 1 - i) + "Z" + "I" * i
            observables.append(SparsePauliOp(pauli_str))
        return observables

    def _build_noise_model(self) -> NoiseModel:
        """
        Build a noise model calibrated to IBM Heron r2 specs.

        Applies active channels only (supports 'depolarizing' and 'readout').
        Amplitude damping and phase damping channels are handled in the PennyLane head;
        the Qiskit head uses depolarizing as the composite noise approximation.

        Returns:
            Qiskit NoiseModel
        """
        noise_model = NoiseModel()

        p1q = self.noise_params.get("p1q", 0.0002)
        p2q = self.noise_params.get("p2q", 0.005)
        readout_error = self.noise_params.get("readout_error", 0.012)

        if "depolarizing" in self.noise_channels or not self.noise_channels:
            # Single-qubit gate depolarizing error
            error_1q = depolarizing_error(p1q, 1)
            noise_model.add_all_qubit_quantum_error(
                error_1q, ["rx", "ry", "rz", "x", "y", "z", "h", "s", "t"]
            )
            # Two-qubit gate depolarizing error
            error_2q = depolarizing_error(p2q, 2)
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cy", "cz", "swap"])

        if "readout" in self.noise_channels or not self.noise_channels:
            ro_error = ReadoutError([
                [1 - readout_error, readout_error],
                [readout_error,     1 - readout_error],
            ])
            noise_model.add_all_qubit_readout_error(ro_error)

        return noise_model

    def _create_qnn(self):
        """
        Create the EstimatorQNN and wrap with TorchConnector.

        Uses StatevectorEstimator (ideal) or AerEstimatorV2 (noisy).
        TorchConnector registers quantum weights as nn.Parameter automatically.

        Gradient strategies (set via self.gradient_method):
          "reverse"    — Adjoint differentiation (ReverseEstimatorGradient). Requires
                         StatevectorEstimator. O(depth) circuit evals, typically 10-30x
                         faster than parameter-shift for circuits with many parameters.
          "spsa"       — Stochastic gradient approximation. O(1) circuit evals per
                         backward step; fastest but introduces gradient noise.
          "param_shift"— Default exact parameter-shift rule. O(2*n_params) evals.
        """
        qc, input_params, weight_params = self._build_circuit()
        observables = self._build_observables()

        if not self.noise:
            # Ideal simulation: StatevectorEstimator (CPU).
            # Note: no GPU variant exists for StatevectorEstimator in Qiskit 1.x;
            # GPU acceleration for ideal circuits is handled by PennyLane
            # (lightning.gpu). Both frameworks are on equal footing here.
            estimator = StatevectorEstimator()
        else:
            noise_model = self._build_noise_model()
            # Noisy simulation: AerEstimatorV2 (CPU by default).
            # GPU path: install qiskit-aer-gpu and set device_preference="gpu".
            # qiskit-aer-gpu exposes the same AerSimulator but compiled with
            # cuStateVec (NVIDIA cuQuantum), giving 10-100x speedup for n>=20 qubits.
            # For n=4 qubits used here, GPU overhead typically outweighs benefit.
            aer_options: Dict[str, Any] = {
                "noise_model": noise_model,
                "shots": self.shots,
            }
            if self.device_preference == "gpu":
                aer_options["device"] = "GPU"
                logger.info("Requesting AerEstimatorV2 on GPU (cuStateVec)")

            try:
                estimator = AerEstimatorV2(options=aer_options)
            except TypeError:
                # Older Aer API: set options post-construction
                estimator = AerEstimatorV2()
                estimator.options.noise_model = noise_model
                estimator.options.default_shots = self.shots
                if self.device_preference == "gpu":
                    try:
                        estimator.options.device = "GPU"
                    except Exception:
                        logger.warning("GPU option not accepted by this Aer version")

        # ------------------------------------------------------------------ #
        # Gradient method selection                                           #
        # ------------------------------------------------------------------ #
        gradient = None  # None → EstimatorQNN uses default param-shift

        if self.gradient_method == "reverse":
            # Adjoint differentiation: single reverse-mode sweep through the
            # statevector — avoids 2*n_params circuit re-runs.
            # Only valid with StatevectorEstimator (ideal, no shots).
            try:
                from qiskit_algorithms.gradients import ReverseEstimatorGradient
                gradient = ReverseEstimatorGradient()
                logger.info("Using ReverseEstimatorGradient (adjoint) for fast gradients")
            except ImportError:
                try:
                    # Older location (qiskit-machine-learning < 0.7)
                    from qiskit.algorithms.gradients import ReverseEstimatorGradient
                    gradient = ReverseEstimatorGradient()
                    logger.info("Using ReverseEstimatorGradient (qiskit.algorithms)")
                except ImportError:
                    logger.warning(
                        "ReverseEstimatorGradient not found (install qiskit-algorithms). "
                        "Falling back to parameter-shift."
                    )
                    gradient = None

        elif self.gradient_method == "spsa":
            # Stochastic gradient: 2 circuit evals per backward step regardless of
            # number of parameters. Fastest option; adds noise to gradient signal.
            try:
                from qiskit_machine_learning.gradients import SPSAEstimatorGradient
                gradient = SPSAEstimatorGradient(estimator, epsilon=0.01, batch_size=1)
                logger.info("Using SPSAEstimatorGradient (stochastic, fast)")
            except (ImportError, Exception) as e:
                logger.warning(f"SPSAEstimatorGradient unavailable ({e}), using param-shift")
                gradient = None

        self.qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            observables=observables,
            input_params=list(input_params),
            weight_params=list(weight_params),
            **({"gradient": gradient} if gradient is not None else {}),
        )

        # TorchConnector registers weights as self.qnn_torch.weight (nn.Parameter)
        init_weights = (0.01 * np.random.randn(self.num_weights)).astype(np.float32)
        self.qnn_torch = TorchConnector(self.qnn, initial_weights=init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid classical-quantum head.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Classical projection + angle encoding: maps features to [-pi/2, pi/2]
        x_proj = torch.tanh(self.proj(x)) * (np.pi / 2)  # (batch_size, n_qubits)

        # Batched QNN call: TorchConnector natively accepts (batch_size, n_qubits).
        # This is significantly faster than looping per sample because Qiskit's
        # StatevectorEstimator batches all PUBs in a single execution.
        q_out_tensor = self.qnn_torch(x_proj)  # (batch_size, n_qubits)

        return self.output(q_out_tensor)        # (batch_size, num_classes)

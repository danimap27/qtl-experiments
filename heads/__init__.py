"""
Heads module: Classification heads for quantum transfer learning.

Factory function to instantiate different head architectures (linear, MLP, quantum).
Supports configuration-driven creation with ablation study overrides.
"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .linear_head import LinearHead
from .mlp_a_head import MLPAHead
from .mlp_b_head import MLPBHead
from .pennylane_head import PennyLaneHead
from .qiskit_head import QiskitHead

logger = logging.getLogger(__name__)


def get_head(
    head_config: Dict[str, Any],
    feature_dim: int,
    num_classes: int,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Factory function to instantiate a classification head.

    Args:
        head_config: Configuration dictionary with keys:
            - name (str): Head type identifier
            - type (str): Head class name (e.g., 'LinearHead', 'MLPAHead', 'PennyLaneHead')
            - n_qubits (int, optional): Number of qubits for quantum heads
            - depth (int, optional): Circuit depth for quantum heads
            - noise (bool, optional): Enable noise for quantum heads
            - noise_channels (list, optional): List of noise channel types to apply
            - noise_params (dict, optional): Noise parameter configuration
            - shots (int, optional): Number of measurement shots for quantum heads
            - hidden_dim (int, optional): Hidden dimension for MLP-A
            - hidden_dims (list, optional): Hidden dimensions for MLP-B
            - activation (str, optional): Activation function name
            - backend (str, optional): Quantum backend (e.g., 'default.qubit')
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        overrides: Optional dictionary to override config values for ablation studies
            - Keys: 'n_qubits', 'depth', 'noise_channels', 'shots'

    Returns:
        Instantiated head module (nn.Module)

    Raises:
        ValueError: If head type is not recognized
    """
    # Apply overrides for ablation studies
    config = head_config.copy()
    if overrides:
        config.update(overrides)

    head_type = config.get("type", "").strip()
    head_name = config.get("name", "").strip()

    if not head_type and not head_name:
        raise ValueError("head_config must contain 'type' or 'name' key")

    logger.info(
        f"Creating head: name={head_name}, type={head_type} "
        f"(feature_dim={feature_dim}, num_classes={num_classes})"
    )

    # Classical heads — matched by type + name from config.yaml
    if head_type == "classical" or head_name in ("linear", "mlp_a", "mlp_b"):
        if head_name == "linear":
            return LinearHead(feature_dim, num_classes)
        elif head_name == "mlp_a":
            hidden_dim = config.get("hidden_dim", 4)
            return MLPAHead(feature_dim, num_classes, hidden_dim=hidden_dim)
        elif head_name == "mlp_b":
            hidden_dims = config.get("hidden_dims", [128, 64])
            return MLPBHead(feature_dim, num_classes, hidden_dims=hidden_dims)
        else:
            raise ValueError(
                f"Unknown classical head name: {head_name}. "
                f"Supported: linear, mlp_a, mlp_b"
            )

    # PennyLane quantum heads
    elif head_type == "pennylane" or head_name.startswith("pl_"):
        n_qubits = config.get("n_qubits", 4)
        depth = config.get("depth", 3)
        backend = config.get("backend", "default.qubit")
        noise = config.get("noise", False)
        noise_channels = config.get("noise_channels", None)
        noise_params = config.get("noise_params", None)
        return PennyLaneHead(
            feature_dim,
            num_classes,
            n_qubits=n_qubits,
            depth=depth,
            backend=backend,
            noise=noise,
            noise_channels=noise_channels,
            noise_params=noise_params,
        )

    # Qiskit quantum heads
    elif head_type == "qiskit" or head_name.startswith("qk_"):
        n_qubits = config.get("n_qubits", 4)
        depth = config.get("depth", 3)
        noise = config.get("noise", False)
        noise_channels = config.get("noise_channels", None)
        noise_params = config.get("noise_params", None)
        shots = config.get("shots", 1024)
        return QiskitHead(
            feature_dim,
            num_classes,
            n_qubits=n_qubits,
            depth=depth,
            noise=noise,
            noise_channels=noise_channels,
            noise_params=noise_params,
            shots=shots,
        )

    else:
        raise ValueError(
            f"Unknown head: name={head_name}, type={head_type}. "
            f"Supported types: classical, pennylane, qiskit. "
            f"Supported names: linear, mlp_a, mlp_b, pl_ideal, pl_noisy, qk_ideal, qk_noisy"
        )


def count_trainable_params(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch module

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

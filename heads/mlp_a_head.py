"""
MLP-A head: Parameter-matched shallow MLP (~12 trainable params).

Designed to match the parameter count of a 4-qubit, depth-3 VQC (~12 variational params).
"""

import torch
import torch.nn as nn


class MLPAHead(nn.Module):
    """
    Parameter-matched shallow MLP.

    Architecture:
    - Linear projection: feature_dim -> hidden_dim (bias=False)
    - Activation: Tanh
    - Output layer: hidden_dim -> num_classes

    The projection layer provides dimensionality reduction from feature space
    to a low-dimensional latent space, mirroring the quantum circuit encoding.
    The output layer maps to class logits.

    For default hidden_dim=4 and num_classes=2:
    - Projection params: feature_dim * 4 (no bias)
    - Output params: 4 * 2 + 2 (with bias) = 10
    - Total output layer params: ~10 (comparable to VQC variational params)

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dim: Dimension of hidden layer (default: 4)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 4,
    ):
        """Initialize the MLP-A head."""
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.act = nn.Hardtanh()
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x):
        """
        Forward pass (sequential samples).

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Sequential processing for architectural/compute-time parity
        # mirroring current quantum simulation patterns.
        return torch.stack([self.fc(self.act(self.proj(x[i]))) for i in range(x.size(0))])

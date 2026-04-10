"""
MLP-B head: Standard deep MLP with configurable hidden dimensions.

Serves as a strong classical baseline for comparisons with quantum heads.
"""

from typing import List

import torch.nn as nn


class MLPBHead(nn.Module):
    """
    Standard deep MLP with multiple hidden layers.

    Default architecture: feature_dim -> 128 -ReLU-> 64 -ReLU-> num_classes

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions (default: [128, 64])
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
    ):
        """Initialize the MLP-B head."""
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = feature_dim

        # Hidden layers with ReLU activations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)

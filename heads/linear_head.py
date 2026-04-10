"""
Linear classification head: Simple linear layer for baseline comparisons.
"""

import torch.nn as nn


class LinearHead(nn.Module):
    """
    Simple linear classification head.

    Maps feature_dim -> num_classes via a single linear layer.

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
    """

    def __init__(self, feature_dim: int, num_classes: int):
        """Initialize the linear head."""
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.fc(x)

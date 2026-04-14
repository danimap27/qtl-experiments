"""
Linear classification head: Simple linear layer for baseline comparisons.
"""

import torch
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
        # Standard dropout for regularization on high-dimensional features
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        """
        Forward pass (sequential samples).
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.dropout(x)
        # Process sample-by-sample for sequential compute-time parity 
        # with inherently sequential quantum simulations.
        return torch.stack([self.fc(x[i]) for i in range(x.size(0))])

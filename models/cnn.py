"""
cnn.py — 1D-CNN baseline for FoG detection.

Architecture (from the report):
  Conv block 1: Conv1D(9→32, k=8), BatchNorm, ReLU, MaxPool(2)
  Conv block 2: Conv1D(32→32, k=8), BatchNorm, ReLU, MaxPool(2)
  FC: 1024 → 128 → 32 → 1 (sigmoid)

Total parameters: ~146k
"""

import torch
import torch.nn as nn


class FoGCNN1D(nn.Module):
    """
    Simple 1D-CNN for binary FoG classification.

    Input:  (B, 128, 9)  — 128-timestep windows, 9 accelerometer channels
    Output: (B, 1)       — FoG probability
    """

    def __init__(
        self,
        in_channels: int = 9,
        conv1_out: int = 32,
        conv2_out: int = 32,
        kernel_size: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_out, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_out, conv2_out, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # Compute the flattened dimension after 2× MaxPool(2) on 128 steps
        # 128 → 64 → 32, so flat_dim = conv2_out * 32
        self._flat_dim = conv2_out * 32

        # FC layers: 1024 (flat_dim) → 128 → 32 → 1
        # The 1024 is the flattened conv output (32 channels × 32 timesteps)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) where T=128, C=9

        Returns:
            logits: (B, 1) — raw logits (apply sigmoid externally or use BCEWithLogitsLoss)
        """
        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)   # (B, 32, 64)
        x = self.conv2(x)   # (B, 32, 32)

        x = x.flatten(1)    # (B, 1024)
        logits = self.classifier(x)  # (B, 1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))

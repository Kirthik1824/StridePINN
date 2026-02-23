"""
conv_ae.py — Convolutional Autoencoder baseline for FoG anomaly detection.

Trained on normal gait only. FoG detected via reconstruction error.

Architecture:
  Encoder: Conv1D(9→32, k=7, pad=3) + BN + ReLU + Pool(2)
           Conv1D(32→32, k=7, pad=3) + BN + ReLU + Pool(2)
           Flatten → Linear(1024→64)
  Decoder: Linear(64→1024) → reshape
           Upsample(2) → Conv1D(32→32, k=7, pad=3) + BN + ReLU
           Upsample(2) → Conv1D(32→9, k=7, pad=3)

Total parameters: ~100k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FoGConvAE(nn.Module):
    """
    Convolutional Autoencoder for gait anomaly detection.

    Input:  (B, 128, 9) — 128-timestep windows, 9 accelerometer channels
    Output: (B, 128, 9) — reconstruction
    """

    def __init__(
        self,
        in_channels: int = 9,
        conv1_out: int = 32,
        conv2_out: int = 32,
        kernel_size: int = 7,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv2_out = conv2_out

        # Use odd kernel size to avoid +1 asymmetry
        pad = kernel_size // 2  # 3 for k=7

        # --- Encoder ---
        self.enc_conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_out, kernel_size, padding=pad),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv1d(conv1_out, conv2_out, kernel_size, padding=pad),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # After 2x MaxPool(2) on 128: 128→64→32, flat = conv2_out * 32
        self._flat_dim = conv2_out * 32

        self.enc_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._flat_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # --- Decoder (mirror of encoder) ---
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, self._flat_dim),
            nn.ReLU(inplace=True),
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv1d(conv2_out, conv1_out, kernel_size, padding=pad),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(inplace=True),
        )
        self.dec_conv2 = nn.Conv1d(conv1_out, in_channels, kernel_size, padding=pad)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.enc_conv1(x)   # (B, 32, 64)
        x = self.enc_conv2(x)   # (B, 32, 32)
        x = x.flatten(1)        # (B, 1024)
        z = self.enc_fc(x)      # (B, latent_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_fc(z)                                     # (B, 1024)
        x = x.view(-1, self.conv2_out, 32)                     # (B, 32, 32)
        x = F.interpolate(x, size=64, mode="nearest")          # (B, 32, 64)
        x = self.dec_conv1(x)                                   # (B, 32, 64)
        x = F.interpolate(x, size=128, mode="nearest")         # (B, 32, 128)
        x = self.dec_conv2(x)                                   # (B, 9, 128)
        x = x.permute(0, 2, 1)                                 # (B, 128, 9)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) where T=128, C=9
        Returns:
            x_hat: (B, T, C) — reconstruction
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window reconstruction error (MSE)."""
        x_hat = self.forward(x)
        # MSE per window: mean over time and channels
        return ((x - x_hat) ** 2).mean(dim=(1, 2))  # (B,)

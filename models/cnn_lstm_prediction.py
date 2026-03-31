"""
models/cnn_lstm_prediction.py — CNN-LSTM for FoG Prediction (Approach 1).

Predicts whether FoG will occur in the next k seconds, using:
  - Raw IMU windows → CNN-LSTM backbone
  - Physics features (FoGI, radius, phase stats) → physics embedding
  - Early Warning Signal features (trends, slopes, entropy) → EWS embedding

The model fuses all three streams before a classifier head.

Input:
  - x: (B, 128, 9) — raw accelerometer windows
  - phys: (B, num_phys_features) — precomputed physics features
  - ews: (B, num_ews_features) — precomputed early-warning signal features

Output: (B, 1) — probability of FoG in next k seconds
"""

import torch
import torch.nn as nn


class FoGCNNLSTMPrediction(nn.Module):
    """
    CNN-LSTM + Physics + EWS hybrid for FoG *prediction*.

    Architecture:
      Conv block 1: Conv1D(9→128, k=4), ReLU, MaxPool(2)
      Conv block 2: Conv1D(128→64, k=4), ReLU, MaxPool(2)
      LSTM: 64 hidden units → temporal representation
      Physics branch: BN → Linear → ReLU (stability features)
      EWS branch: BN → Linear → ReLU (early-warning features)
      Fusion: cat(LSTM_hidden, phys_embed, ews_embed) → FC → 1
    """

    def __init__(
        self,
        in_channels: int = 9,
        conv1_out: int = 128,
        conv2_out: int = 64,
        kernel_size: int = 4,
        lstm_hidden: int = 64,
        num_phys_features: int = 9,
        num_ews_features: int = 5,
        phys_embed_dim: int = 16,
        ews_embed_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ----- CNN backbone -----
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_out, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_out, conv2_out, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # ----- LSTM -----
        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ----- Physics feature branch -----
        self.phys_bn = nn.BatchNorm1d(num_phys_features)
        self.phys_embed = nn.Sequential(
            nn.Linear(num_phys_features, phys_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ----- EWS feature branch -----
        self.ews_bn = nn.BatchNorm1d(num_ews_features)
        self.ews_embed = nn.Sequential(
            nn.Linear(num_ews_features, ews_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ----- Fusion classifier -----
        fused_dim = lstm_hidden + phys_embed_dim + ews_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor, phys: torch.Tensor, ews: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, C) where T=128, C=9
            phys: (B, num_phys_features)
            ews:  (B, num_ews_features)

        Returns:
            logits: (B, 1)
        """
        # CNN: (B, T, C) → (B, C, T) for Conv1d
        h = x.permute(0, 2, 1)
        h = self.conv1(h)   # (B, 128, 64)
        h = self.conv2(h)   # (B, 64, 32)

        # LSTM
        h = h.permute(0, 2, 1)  # (B, 32, 64)
        _, (h_n, _) = self.lstm(h)
        lstm_out = self.lstm_dropout(h_n[-1])  # (B, 64)

        # Physics branch
        phys_out = self.phys_bn(phys)
        phys_out = self.phys_embed(phys_out)  # (B, phys_embed_dim)

        # EWS branch
        ews_out = self.ews_bn(ews)
        ews_out = self.ews_embed(ews_out)  # (B, ews_embed_dim)

        # Fusion
        fused = torch.cat([lstm_out, phys_out, ews_out], dim=1)
        logits = self.classifier(fused)

        return logits

    def predict_proba(
        self, x: torch.Tensor, phys: torch.Tensor, ews: torch.Tensor
    ) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x, phys, ews))

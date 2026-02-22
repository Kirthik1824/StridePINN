"""
cnn_lstm.py — CNN-LSTM baseline for FoG detection.

Architecture (from the report):
  Conv block 1: Conv1D(9→128, k=4), ReLU, MaxPool(2)
  Conv block 2: Conv1D(128→64, k=4), ReLU, MaxPool(2)
  LSTM: 64 hidden units, recurrent dropout
  FC: 64 → 80 → 40 → 1 (sigmoid)

Total parameters: ~79k
"""

import torch
import torch.nn as nn


class FoGCNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid for binary FoG classification.

    Input:  (B, 128, 9)  — 128-timestep windows, 9 accelerometer channels
    Output: (B, 1)       — FoG probability
    """

    def __init__(
        self,
        in_channels: int = 9,
        conv1_out: int = 128,
        conv2_out: int = 64,
        kernel_size: int = 4,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1_out, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_out, conv2_out, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # LSTM — processes the conv features as a sequence
        # After 2× MaxPool(2) on 128: seq_len = 32
        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # single layer, so dropout param is ignored
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # FC head: 64 → 80 → 40 → 1
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(80, 40),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(40, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) where T=128, C=9

        Returns:
            logits: (B, 1) — raw logits
        """
        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)   # (B, 128, 64)
        x = self.conv2(x)   # (B, 64, 32)

        # Reshape for LSTM: (B, seq_len, features)
        x = x.permute(0, 2, 1)  # (B, 32, 64)

        # LSTM — take the last hidden state
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (B, 32, 64)
        x = self.lstm_dropout(h_n[-1])     # (B, 64)

        logits = self.classifier(x)  # (B, 1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))

"""
config.py — Central configuration for StridePINN.

All hyperparameters, paths, and preprocessing constants live here
so that every script imports a single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # ----------------------------------------------------------------
    #  Paths
    # ----------------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parent
    raw_data_dir: Path = field(default=None)
    processed_data_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    checkpoint_dir: Path = field(default=None)

    def __post_init__(self):
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"
        self.results_dir = self.project_root / "results"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.figures_dir = self.results_dir / "figures"

    # ----------------------------------------------------------------
    #  Dataset
    # ----------------------------------------------------------------
    num_subjects: int = 10
    # Sensor channels: 1 ankle (3 axes) + 1 thigh (3 axes) + trunk (3 axes)
    num_channels: int = 9
    original_fs: int = 64       # Hz — Daphnet native sampling rate
    target_fs: int = 40         # Hz — after resampling

    # ----------------------------------------------------------------
    #  Preprocessing
    # ----------------------------------------------------------------
    # Anti-alias filter before downsampling
    antialias_order: int = 8
    antialias_cutoff: float = 20.0  # Hz (Nyquist of target_fs)

    # Band-pass filter
    bandpass_order: int = 4
    bandpass_low: float = 0.3       # Hz
    bandpass_high: float = 15.0     # Hz

    # Sliding window
    window_size: int = 128          # samples  (3.2 s @ 40 Hz)
    window_stride: int = 32         # samples  (0.8 s — 75 % overlap)

    # Window labelling thresholds
    fog_label_threshold: float = 0.5   # >50 % FoG → label 1
    normal_label_threshold: float = 0.1  # <10 % FoG → label 0

    # ----------------------------------------------------------------
    #  Data Augmentation (supervised baselines)
    # ----------------------------------------------------------------
    fog_oversample_factor: int = 3
    jitter_samples: int = 4         # ±0.1 s at 40 Hz

    # ----------------------------------------------------------------
    #  Model — 1D-CNN
    # ----------------------------------------------------------------
    cnn_conv1_out: int = 32
    cnn_conv2_out: int = 32
    cnn_kernel_size: int = 8
    cnn_fc_sizes: Tuple[int, ...] = (1024, 128, 32)
    cnn_dropout: float = 0.3

    # ----------------------------------------------------------------
    #  Model — CNN-LSTM
    # ----------------------------------------------------------------
    lstm_conv1_out: int = 128
    lstm_conv2_out: int = 64
    lstm_kernel_size: int = 4
    lstm_hidden: int = 64
    lstm_fc_sizes: Tuple[int, ...] = (80, 40)
    lstm_dropout: float = 0.3

    # ----------------------------------------------------------------
    #  Model — PINN
    # ----------------------------------------------------------------
    latent_dim: int = 2
    encoder_hidden: int = 64
    ode_hidden: int = 32
    decoder_out: int = 3            # reconstruct ankle (3 axes)

    # Physics loss weights
    lambda_cyc: float = 1.0
    lambda_phi: float = 10.0
    lambda_smooth: float = 0.1

    # ODE solver
    ode_method: str = "rk4"         # Switched from dopri5 for speed
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    ode_step_size: float = 0.1      # Fixed step size for solvers like rk4

    # ----------------------------------------------------------------
    #  Training
    # ----------------------------------------------------------------
    batch_size: int = 1024          # Increased from 128 to saturate GPU
    num_epochs: int = 50            # Reduced from 100 (PINN converges fast)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    cosine_t_max: int = 50
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10   # Tighter patience

    # ----------------------------------------------------------------
    #  Reproducibility
    # ----------------------------------------------------------------
    seed: int = 42


# Global singleton
cfg = Config()

"""
utils.py — Shared helpers for StridePINN.

Seed setting, device selection, logging, checkpoint I/O, metric helpers.
"""

import random
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)


# -----------------------------------------------------------------
#  Reproducibility
# -----------------------------------------------------------------
def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------
#  Device
# -----------------------------------------------------------------
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------
#  Logging
# -----------------------------------------------------------------
def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """Create a console (+ optional file) logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -----------------------------------------------------------------
#  Checkpointing
# -----------------------------------------------------------------
def save_checkpoint(state: dict, path: Path):
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device = None) -> dict:
    """Load a training checkpoint."""
    return torch.load(path, map_location=device, weights_only=False)


# -----------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics from ground-truth labels and predicted
    probabilities (or anomaly scores).

    Returns dict with keys: auc, f1, sensitivity, specificity, precision.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "auc": auc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the threshold that maximises Youden's J statistic
    (sensitivity + specificity − 1) on the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def compute_detection_latency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_stride_sec: float = 0.8,
) -> dict:
    """
    Compute detection latency for true-positive FoG episodes.

    A FoG episode is a contiguous run of y_true==1 windows.
    Latency = number of windows from episode start to first correct
    detection × window_stride_sec.

    Returns dict with median, q25, q75, and list of latencies.
    """
    latencies = []
    in_episode = False
    episode_start = None

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_episode:
            in_episode = True
            episode_start = i
        elif y_true[i] == 0 and in_episode:
            in_episode = False

        if in_episode and y_pred[i] == 1 and episode_start is not None:
            latency = (i - episode_start) * window_stride_sec
            latencies.append(latency)
            in_episode = False  # count only first detection per episode
            episode_start = None

    if len(latencies) == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "all": []}

    arr = np.array(latencies)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "all": latencies,
    }

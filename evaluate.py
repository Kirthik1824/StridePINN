"""
evaluate.py — Evaluation utilities for all three models.

Provides:
  - evaluate_baseline(): per-fold eval for CNN/CNN-LSTM
  - evaluate_pinn(): per-fold eval with residual + phase anomaly scoring
  - compare_models(): load saved results and produce comparison tables
"""

import json
from pathlib import Path

import numpy as np
import torch

from config import cfg
from utils import compute_metrics, find_optimal_threshold, compute_detection_latency
from models.pinn import GaitPINN
from data.dataset import GaitDataset, get_loso_splits, normalise_fold, apply_normalisation


def load_results(model_name: str) -> list:
    """Load saved per-fold results for a model."""
    path = cfg.results_dir / f"{model_name}_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found at {path}. Run training first.")
    with open(path) as f:
        return json.load(f)


def compare_models(model_names: list = None) -> dict:
    """
    Load and compare results across models.

    Returns dict[model_name] = dict of mean ± std for each metric.
    """
    if model_names is None:
        model_names = ["cnn", "cnn_lstm", "pinn"]

    comparison = {}

    for name in model_names:
        try:
            results = load_results(name)
        except FileNotFoundError:
            print(f"Skipping {name}: results not found")
            continue

        metrics_keys = ["auc", "sensitivity", "specificity", "f1"]
        stats = {}

        for key in metrics_keys:
            values = [r["test_metrics"][key] for r in results]
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

        latencies = [
            r["latency"]["median"]
            for r in results
            if r.get("latency") and not np.isnan(r["latency"]["median"])
        ]
        if latencies:
            stats["latency"] = {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
            }

        comparison[name] = stats

    return comparison


def print_comparison_table(comparison: dict):
    """Print a formatted comparison table."""
    print(f"\n{'Model':15s} | {'AUC':12s} | {'Sensitivity':12s} | {'Specificity':12s} | {'F1':12s} | {'Latency (s)':12s}")
    print("-" * 85)

    for name, stats in comparison.items():
        row = f"{name:15s}"
        for key in ["auc", "sensitivity", "specificity", "f1"]:
            if key in stats:
                row += f" | {stats[key]['mean']:.3f}±{stats[key]['std']:.3f}"
            else:
                row += f" | {'N/A':>12s}"

        if "latency" in stats:
            row += f" | {stats['latency']['mean']:.2f}±{stats['latency']['std']:.2f}"
        else:
            row += f" | {'N/A':>12s}"

        print(row)


def evaluate_pinn_single_window(
    model: GaitPINN,
    window: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Evaluate a single window through the PINN and return detailed diagnostics.

    Args:
        model: trained GaitPINN
        window: (1, 128, 9) or (128, 9)
        device: torch device

    Returns:
        dict with z_traj, residual trace, phase trace, anomaly scores
    """
    if window.dim() == 2:
        window = window.unsqueeze(0)

    model.eval()
    window = window.to(device)

    with torch.no_grad():
        out = model.forward(window)
        scores = model.compute_anomaly_scores(window)

    return {
        "z_traj": out["z_traj"].cpu().numpy(),       # (T, 1, 2)
        "x_hat": out["x_hat"].cpu().numpy(),          # (T, 1, 3)
        "residual": scores["residual_all"].cpu().numpy(),  # (1, T-2)
        "residual_max": scores["residual_max"].item(),
        "phase": scores["phase_all"].cpu().numpy(),   # (1, T)
        "phase_advance": scores["phase_advance"].item(),
    }


if __name__ == "__main__":
    comparison = compare_models()
    print_comparison_table(comparison)

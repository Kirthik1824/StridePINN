"""
visualization/rule_plots.py — Visualization for Approach 2: Rule-Based Detection.

Generates:
  1. Threshold sweep analysis
  2. ROC + PR curves
  3. Comparison table (Rule-based vs CNN-LSTM)
  4. Per-subject performance heatmap
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from data.dataset import GaitDataset, get_loso_splits
from rule_detector import RuleBasedDetector


def plot_threshold_sweep(sweep_data: dict = None, fold: int = 1):
    """Plot recall, precision, F1 vs detection threshold."""
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if sweep_data is None:
        sweep_path = cfg.results_dir / "rule_detector_sweep.json"
        if not sweep_path.exists():
            print("No sweep data found. Run with --sweep first.")
            return
        with open(sweep_path) as f:
            sweep_data = json.load(f)

    key = f"fold_{fold}"
    if key not in sweep_data or not sweep_data[key]:
        print(f"No sweep data for {key}")
        return

    data = sweep_data[key]
    thresholds = [d["threshold"] for d in data]
    recalls = [d["recall"] for d in data]
    precisions = [d["precision"] for d in data]
    f1s = [d["f1"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, recalls, color="#F44336", linewidth=2, label="Recall (Sensitivity)")
    ax.plot(thresholds, precisions, color="#2196F3", linewidth=2, label="Precision")
    ax.plot(thresholds, f1s, color="#4CAF50", linewidth=2, label="F1 Score")

    # Mark optimal F1
    best_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_idx], color="#333", linestyle="--", alpha=0.5, label=f"Optimal θ={thresholds[best_idx]:.2f}")

    ax.set_xlabel("Detection Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Sweep — Fold {fold}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = fig_dir / f"threshold_sweep_fold{fold}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_roc_pr_curves(results_path: Path = None):
    """Plot ROC and PR curves from saved results."""
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if results_path is None:
        results_path = cfg.results_dir / "rule_detector_results.json"

    if not results_path.exists():
        print("No results found. Run rule_detector first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Rule-Based Detector — ROC & PR Curves", fontsize=14, fontweight="bold")

    for r in all_results:
        metrics = r["test_metrics"]
        fold = r["fold"]

        # Plot point on ROC space
        fpr = 1 - metrics["specificity"]
        tpr = metrics["sensitivity"]
        ax1.scatter(fpr, tpr, s=60, alpha=0.7, zorder=5)
        ax1.annotate(f"S{fold}", (fpr, tpr), fontsize=8, ha="center", va="bottom")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("ROC Space")
    ax1.grid(True, alpha=0.3)

    # PR space
    for r in all_results:
        metrics = r["test_metrics"]
        fold = r["fold"]
        ax2.scatter(metrics["sensitivity"], metrics["precision"], s=60, alpha=0.7, zorder=5)
        ax2.annotate(f"S{fold}", (metrics["sensitivity"], metrics["precision"]),
                     fontsize=8, ha="center", va="bottom")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Space")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "rule_roc_pr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_subject_heatmap(results_path: Path = None):
    """Plot per-subject performance heatmap."""
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if results_path is None:
        results_path = cfg.results_dir / "rule_detector_results.json"

    if not results_path.exists():
        print("No results found.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    metrics_keys = ["auc", "f1", "sensitivity", "specificity", "precision"]
    subjects = [r["test_subject"] for r in all_results]
    data = np.array([
        [r["test_metrics"][k] for k in metrics_keys]
        for r in all_results
    ])

    fig, ax = plt.subplots(figsize=(10, max(4, len(subjects) * 0.5 + 2)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics_keys)))
    ax.set_xticklabels([k.capitalize() for k in metrics_keys])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"Subject {s}" for s in subjects])

    # Add value annotations
    for i in range(len(subjects)):
        for j in range(len(metrics_keys)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Per-Subject Performance — Rule-Based Detector", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = fig_dir / "rule_subject_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_all_rule_plots():
    """Generate all Approach 2 visualization plots."""
    plot_roc_pr_curves()
    plot_per_subject_heatmap()

    # Try threshold sweep if available
    sweep_path = cfg.results_dir / "rule_detector_sweep.json"
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep_data = json.load(f)
        for key in sweep_data:
            fold = int(key.split("_")[1])
            plot_threshold_sweep(sweep_data, fold)
            break  # Just plot first fold as example

    print("\nAll rule-based plots generated.")


if __name__ == "__main__":
    generate_all_rule_plots()

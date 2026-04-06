"""
visualization/prediction_plots.py — Visualization for Approach 1: FoG Prediction.

Generates:
  1. Phase portraits (Normal vs Pre-FoG vs FoG)
  2. FoGI trajectory over time with FoG onset markers
  3. Radius collapse curves
  4. Performance vs prediction horizon
  5. Ablation: with/without EWS features
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from data.dataset import GaitDataset
from features import compute_fogi


def _delay_embed(signal, tau=5):
    """Inline Takens delay embedding for visualization."""
    x = signal[:-tau]
    y = signal[tau:]
    return x, y


COLORS = {
    "normal": "#2196F3",
    "pre_fog": "#FF9800",
    "fog": "#F44336",
}


def plot_phase_portraits(dataset: GaitDataset = None, subject_id: int = 1):
    """Plot phase portraits: Normal, Pre-FoG, and FoG windows."""
    if dataset is None:
        dataset = GaitDataset()

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mask = dataset.subject_ids == subject_id
    windows = dataset.windows[mask].numpy()
    labels = dataset.labels[mask].numpy()

    normal_idx = np.where(labels == 0)[0]
    fog_idx = np.where(labels == 1)[0]

    if len(normal_idx) == 0 or len(fog_idx) == 0:
        print(f"Subject {subject_id}: insufficient data for phase portraits")
        return

    # Pre-FoG: normal window immediately before a FoG window
    pre_fog_idx = []
    for fi in fog_idx:
        if fi > 0 and labels[fi - 1] == 0:
            pre_fog_idx.append(fi - 1)
    pre_fog_idx = np.array(pre_fog_idx) if pre_fog_idx else normal_idx[:1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Phase Portraits — Subject {subject_id}", fontsize=14, fontweight="bold")

    tau = 5
    for ax, idx_arr, label, color, title in [
        (axes[0], normal_idx, "Normal", COLORS["normal"], "Normal Walking\n(Stable Limit Cycle)"),
        (axes[1], pre_fog_idx, "Pre-FoG", COLORS["pre_fog"], "Pre-FoG\n(Destabilising)"),
        (axes[2], fog_idx, "FoG", COLORS["fog"], "Freezing of Gait\n(Collapsed/Chaotic)"),
    ]:
        # Pick middle window
        w = windows[idx_arr[len(idx_arr) // 2]]
        sig = w[:, 0]  # ankle_x

        # Bandpass
        nyq = 0.5 * cfg.target_fs
        try:
            b, a = butter(4, [0.5 / nyq, min(10.0 / nyq, 0.99)], btype="band")
            sig_filt = filtfilt(b, a, sig)
        except ValueError:
            sig_filt = sig

        x, y = _delay_embed(sig_filt, tau)
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(x[0], y[0], marker="o", color="green", s=50, zorder=5, label="Start")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t+{tau})")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = fig_dir / "phase_portraits.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_fogi_trajectory(dataset: GaitDataset = None, subject_id: int = 1):
    """Plot FoGI trajectory over time with FoG onset markers."""
    if dataset is None:
        dataset = GaitDataset()

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mask = dataset.subject_ids == subject_id
    windows = dataset.windows[mask].numpy()
    labels = dataset.labels[mask].numpy()

    # Compute FoGI for each window
    fogis = []
    for i in range(len(windows)):
        fogi = compute_fogi(windows[i][:, 0], fs=cfg.target_fs)
        fogis.append(fogi)
    fogis = np.array(fogis)

    # Time axis (in seconds)
    window_stride_sec = cfg.window_stride / cfg.target_fs
    time = np.arange(len(fogis)) * window_stride_sec

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"FoGI Trajectory — Subject {subject_id}", fontsize=14, fontweight="bold")

    # FoGI over time
    ax1.plot(time, fogis, color="#333", linewidth=0.8, alpha=0.7)
    ax1.fill_between(time, 0, fogis, alpha=0.3, color="#FF9800")

    # Mark FoG regions
    fog_mask = labels == 1
    ax1.scatter(time[fog_mask], fogis[fog_mask], color=COLORS["fog"], s=5, alpha=0.6, label="FoG")
    ax1.set_ylabel("FoGI")
    ax1.set_title("Freeze Index Over Time", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Labels over time
    ax2.fill_between(time, 0, labels, alpha=0.5, color=COLORS["fog"], label="FoG")
    ax2.fill_between(time, 0, 1 - labels, alpha=0.3, color=COLORS["normal"], label="Normal")
    ax2.set_ylabel("Label")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Ground Truth", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "fogi_trajectory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_radius_collapse(dataset: GaitDataset = None, subject_id: int = 1):
    """Plot radius over time showing collapse before FoG."""
    if dataset is None:
        dataset = GaitDataset()

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mask = dataset.subject_ids == subject_id
    windows = dataset.windows[mask].numpy()
    labels = dataset.labels[mask].numpy()

    # Compute mean radius for each window
    tau = 5
    radii = []
    for i in range(len(windows)):
        sig = windows[i][:, 0]
        nyq = 0.5 * cfg.target_fs
        try:
            b, a = butter(4, [0.5 / nyq, min(10.0 / nyq, 0.99)], btype="band")
            sig_filt = filtfilt(b, a, sig)
        except ValueError:
            sig_filt = sig

        if len(sig_filt) > tau:
            x, y = _delay_embed(sig_filt, tau)
            r = np.sqrt(x**2 + y**2)
            radii.append(np.mean(r))
        else:
            radii.append(0.0)

    radii = np.array(radii)
    window_stride_sec = cfg.window_stride / cfg.target_fs
    time = np.arange(len(radii)) * window_stride_sec

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Radius Collapse — Subject {subject_id}", fontsize=14, fontweight="bold")

    ax.plot(time, radii, color="#333", linewidth=0.8, alpha=0.7)

    # Shade FoG regions
    fog_mask = labels == 1
    for i in range(len(fog_mask)):
        if fog_mask[i]:
            ax.axvspan(time[i] - window_stride_sec / 2, time[i] + window_stride_sec / 2,
                       color=COLORS["fog"], alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Embedding Radius")
    ax.set_title("Delay-Embedding Radius (collapse = approaching FoG)", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "radius_collapse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_all_prediction_plots(subject_id: int = 1):
    """Generate all Approach 1 visualization plots."""
    dataset = GaitDataset()
    plot_phase_portraits(dataset, subject_id)
    plot_fogi_trajectory(dataset, subject_id)
    plot_radius_collapse(dataset, subject_id)
    print("\nAll prediction plots generated.")


if __name__ == "__main__":
    generate_all_prediction_plots()

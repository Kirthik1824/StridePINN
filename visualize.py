"""
visualize.py — Visualization suite for StridePINN.

Generates:
  1. 2-D latent-space limit-cycle plots (normal vs. FoG)
  2. Dynamics residual r(t) time-traces
  3. Phase φ(t) progression showing stagnation during FoG
  4. ROC curves with confidence bands
  5. Per-subject performance bar charts
  6. Loss convergence curves

All plots saved to results/figures/.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from config import cfg
from utils import get_device, compute_metrics
from models.pinn import GaitPINN
from data.dataset import GaitDataset, get_loso_splits, normalise_fold, apply_normalisation


# Style setup
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def ensure_fig_dir():
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# -----------------------------------------------------------------
#  1. Latent-space limit-cycle plot
# -----------------------------------------------------------------
def plot_latent_trajectories(
    model: GaitPINN,
    normal_windows: torch.Tensor,
    fog_windows: torch.Tensor = None,
    device: torch.device = None,
    n_samples: int = 20,
    title: str = "Latent Space Trajectories",
    save_name: str = "latent_trajectories.png",
):
    """
    Plot latent-space trajectories for normal and FoG windows.

    Normal gait should trace closed loops (limit cycles).
    FoG windows should show departures / collapses.
    """
    fig_dir = ensure_fig_dir()
    device = device or get_device()
    model.eval()

    fig, axes = plt.subplots(1, 2 if fog_windows is not None else 1,
                              figsize=(14 if fog_windows is not None else 7, 6))
    if fog_windows is None:
        axes = [axes]

    # Plot normal trajectories
    ax = axes[0]
    indices = np.random.choice(len(normal_windows), min(n_samples, len(normal_windows)), replace=False)

    for idx in indices:
        window = normal_windows[idx:idx+1].to(device)
        with torch.no_grad():
            out = model.forward(window)
            z = out["z_traj"][:, 0, :].cpu().numpy()  # (T, 2)

        ax.plot(z[:, 0], z[:, 1], alpha=0.4, linewidth=1.0, color="steelblue")
        ax.scatter(z[0, 0], z[0, 1], s=15, color="green", zorder=5)   # start
        ax.scatter(z[-1, 0], z[-1, 1], s=15, color="red", zorder=5, marker="x")  # end

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Normal Gait — Limit Cycles")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot FoG trajectories
    if fog_windows is not None and len(fog_windows) > 0:
        ax = axes[1]
        indices = np.random.choice(len(fog_windows), min(n_samples, len(fog_windows)), replace=False)

        for idx in indices:
            window = fog_windows[idx:idx+1].to(device)
            with torch.no_grad():
                out = model.forward(window)
                z = out["z_traj"][:, 0, :].cpu().numpy()

            ax.plot(z[:, 0], z[:, 1], alpha=0.4, linewidth=1.0, color="tomato")
            ax.scatter(z[0, 0], z[0, 1], s=15, color="green", zorder=5)
            ax.scatter(z[-1, 0], z[-1, 1], s=15, color="red", zorder=5, marker="x")

        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_title("FoG Windows — Disrupted Dynamics")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_dir / save_name}")


# -----------------------------------------------------------------
#  2. Dynamics residual time-trace
# -----------------------------------------------------------------
def plot_residual_traces(
    model: GaitPINN,
    normal_windows: torch.Tensor,
    fog_windows: torch.Tensor,
    device: torch.device = None,
    n_samples: int = 5,
    save_name: str = "residual_traces.png",
):
    """Plot r(t) for normal vs FoG windows side by side."""
    fig_dir = ensure_fig_dir()
    device = device or get_device()
    model.eval()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    dt = 3.2 / 128  # seconds per timestep
    t_axis = np.arange(1, 127) * dt  # interior points

    # Normal
    ax = axes[0]
    for i in range(min(n_samples, len(normal_windows))):
        window = normal_windows[i:i+1].to(device)
        with torch.no_grad():
            scores = model.compute_anomaly_scores(window)
            r = scores["residual_all"][0].cpu().numpy()
        ax.plot(t_axis[:len(r)], r, alpha=0.6, linewidth=1.0)

    ax.set_ylabel("Residual $r(t)$")
    ax.set_title("Normal Gait — Low Residuals")
    ax.grid(True, alpha=0.3)

    # FoG
    ax = axes[1]
    for i in range(min(n_samples, len(fog_windows))):
        window = fog_windows[i:i+1].to(device)
        with torch.no_grad():
            scores = model.compute_anomaly_scores(window)
            r = scores["residual_all"][0].cpu().numpy()
        ax.plot(t_axis[:len(r)], r, alpha=0.6, linewidth=1.0, color="tomato")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual $r(t)$")
    ax.set_title("FoG — Residual Spikes at Freeze Onset")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Dynamics Residual Time-Traces", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_dir / save_name}")


# -----------------------------------------------------------------
#  3. Phase progression plot
# -----------------------------------------------------------------
def plot_phase_progression(
    model: GaitPINN,
    normal_windows: torch.Tensor,
    fog_windows: torch.Tensor,
    device: torch.device = None,
    n_samples: int = 5,
    save_name: str = "phase_progression.png",
):
    """Plot φ(t) showing steady advancing for normal and stagnation for FoG."""
    fig_dir = ensure_fig_dir()
    device = device or get_device()
    model.eval()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t_axis = np.linspace(0, 3.2, 128)

    # Normal
    ax = axes[0]
    for i in range(min(n_samples, len(normal_windows))):
        window = normal_windows[i:i+1].to(device)
        with torch.no_grad():
            scores = model.compute_anomaly_scores(window)
            phase = scores["phase_all"][0].cpu().numpy()

        # Unwrap for continuous plot
        phase_unwrapped = np.unwrap(phase)
        ax.plot(t_axis, phase_unwrapped, alpha=0.6, linewidth=1.0)

    ax.set_ylabel("Phase $\\phi(t)$ (rad)")
    ax.set_title("Normal Gait — Steady Phase Advance")
    ax.grid(True, alpha=0.3)

    # FoG
    ax = axes[1]
    for i in range(min(n_samples, len(fog_windows))):
        window = fog_windows[i:i+1].to(device)
        with torch.no_grad():
            scores = model.compute_anomaly_scores(window)
            phase = scores["phase_all"][0].cpu().numpy()

        phase_unwrapped = np.unwrap(phase)
        ax.plot(t_axis, phase_unwrapped, alpha=0.6, linewidth=1.0, color="tomato")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase $\\phi(t)$ (rad)")
    ax.set_title("FoG — Phase Stagnation (Plateau)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Latent Phase Progression", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_dir / save_name}")


# -----------------------------------------------------------------
#  4. ROC curves
# -----------------------------------------------------------------
def plot_roc_curves(save_name: str = "roc_curves.png"):
    """Plot ROC curves for all models from saved results."""
    from sklearn.metrics import roc_curve, auc

    fig_dir = ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {"cnn": "steelblue", "cnn_lstm": "forestgreen", "pinn": "crimson"}
    labels = {"cnn": "1D-CNN", "cnn_lstm": "CNN-LSTM", "pinn": "PINN"}

    for model_name in ["cnn", "cnn_lstm", "pinn"]:
        results_path = cfg.results_dir / f"{model_name}_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        aucs = [r["test_metrics"]["auc"] for r in results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Plot mean performance point
        sens_vals = [r["test_metrics"]["sensitivity"] for r in results]
        spec_vals = [r["test_metrics"]["specificity"] for r in results]
        mean_fpr = 1 - np.mean(spec_vals)
        mean_tpr = np.mean(sens_vals)

        ax.scatter(
            mean_fpr, mean_tpr,
            s=120, color=colors[model_name], zorder=5,
            label=f"{labels[model_name]} (AUC={mean_auc:.3f}±{std_auc:.3f})",
            edgecolors="black", linewidth=0.5,
        )

        # Error bars
        ax.errorbar(
            mean_fpr, mean_tpr,
            xerr=np.std([1 - s for s in spec_vals]),
            yerr=np.std(sens_vals),
            color=colors[model_name], alpha=0.4, fmt="none",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Model Comparison — ROC Space")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_dir / save_name}")


# -----------------------------------------------------------------
#  5. Per-subject performance
# -----------------------------------------------------------------
def plot_per_subject_performance(save_name: str = "per_subject_performance.png"):
    """Bar chart of per-fold (per-subject) F1 for each model."""
    fig_dir = ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = ["cnn", "cnn_lstm", "pinn"]
    colors = ["steelblue", "forestgreen", "crimson"]
    bar_width = 0.25

    for mi, (model_name, color) in enumerate(zip(model_names, colors)):
        results_path = cfg.results_dir / f"{model_name}_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        folds = [r["fold"] for r in results]
        f1_vals = [r["test_metrics"]["f1"] for r in results]

        x = np.arange(len(folds)) + mi * bar_width
        ax.bar(x, f1_vals, bar_width, label=model_name.upper().replace("_", "-"),
               color=color, alpha=0.8)

    ax.set_xlabel("Test Subject (Fold)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Subject F1 Score Comparison")
    ax.set_xticks(np.arange(len(folds)) + bar_width)
    ax.set_xticklabels([f"S{f:02d}" for f in folds])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_dir / save_name}")


# -----------------------------------------------------------------
#  Main — generate all visualizations
# -----------------------------------------------------------------
def generate_all_visualizations(fold: int = 1):
    """Generate all visualization plots for a specified fold."""
    fig_dir = ensure_fig_dir()
    device = get_device()
    print(f"\nGenerating visualizations (fold={fold})...")

    # Try to load PINN model for latent-space plots
    pinn_ckpt = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    if pinn_ckpt.exists():
        print("  Loading PINN checkpoint...")
        ckpt = torch.load(pinn_ckpt, map_location=device, weights_only=False)

        model = GaitPINN(
            in_dim=cfg.num_channels,
            latent_dim=cfg.latent_dim,
            encoder_hidden=cfg.encoder_hidden,
            ode_hidden=cfg.ode_hidden,
            decoder_out=cfg.decoder_out,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Load dataset and get fold data
        dataset = GaitDataset()
        mean, std = ckpt["norm_mean"], ckpt["norm_std"]

        # Find test subject windows
        test_mask = dataset.subject_ids == fold
        test_w = apply_normalisation(dataset.windows[test_mask].to(device), mean, std)
        test_l = dataset.labels[test_mask]

        normal_w = test_w[test_l == 0]
        fog_w = test_w[test_l == 1]

        print(f"  Test windows: {len(normal_w)} normal, {len(fog_w)} FoG")

        # Generate PINN-specific plots
        plot_latent_trajectories(model, normal_w, fog_w, device)
        if len(fog_w) > 0:
            plot_residual_traces(model, normal_w, fog_w, device)
            plot_phase_progression(model, normal_w, fog_w, device)
    else:
        print(f"  PINN checkpoint not found at {pinn_ckpt}, skipping latent-space plots.")

    # Generate comparison plots from saved results
    plot_roc_curves()
    plot_per_subject_performance()

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    args = p.parse_args()
    generate_all_visualizations(args.fold)

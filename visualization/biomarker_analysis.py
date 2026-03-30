"""
biomarker_analysis.py — Publication-ready interpretability figures for StridePINN.

Generates:
  1. Biomarker alignment plot: overlay FoG annotations with residual + phase traces
  2. Limit cycle portrait: normal vs FoG trajectories in latent phase plane
  3. Dynamics summary: per-subject AUC with biomarker panels

All plots are designed for IEEE two-column format.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from config import cfg
from models.pinn import GaitPINN
from data.dataset import GaitDataset, apply_per_subject_normalization
from utils import get_device, load_checkpoint


# IEEE-friendly style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load_model_and_data(fold=1):
    """Load trained PINN and dataset for a given fold."""
    device = get_device()
    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train the PINN first.")

    ckpt = load_checkpoint(ckpt_path, device=device)

    model = GaitPINN(
        in_dim=cfg.num_channels,
        latent_dim=cfg.latent_dim,
        encoder_hidden=cfg.encoder_hidden,
        ode_hidden=cfg.ode_hidden,
        decoder_out=cfg.decoder_out,
        ode_mode=cfg.ode_mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = GaitDataset()
    
    # Get test subject data
    mask = dataset.subject_ids == fold
    # Use per-subject normalization (sync with trainer)
    windows = apply_per_subject_normalization(dataset.windows[mask], dataset.subject_ids[mask]).to(device)
    labels = dataset.labels[mask].numpy()

    return model, windows, labels, ckpt, device


def plot_biomarker_alignment(fold=1, subject_id=None):
    """
    Publication figure: Three-panel biomarker alignment plot.

    Panel 1: Ground-truth FoG annotations
    Panel 2: Dynamics residual r_max(t) with threshold
    Panel 3: Phase advance ΔΦ with threshold

    Shows temporal alignment between biomarkers and FoG episodes.
    """
    if subject_id is None:
        subject_id = fold

    model, windows, labels, ckpt, device = load_model_and_data(fold)

    # Compute scores in batches
    all_residuals, all_phases = [], []
    with torch.no_grad():
        for i in range(0, len(windows), 128):
            batch = windows[i:i + 128]
            scores = model.compute_anomaly_scores(batch)
            all_residuals.append(scores["residual_max"].cpu().numpy())
            all_phases.append(scores["phase_advance"].cpu().numpy())

    residuals = np.concatenate(all_residuals)
    phases = np.concatenate(all_phases)
    time_sec = np.arange(len(labels)) * 0.8  # window stride = 0.8s

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2, 2]})

    # Panel 1: Ground truth FoG
    ax = axes[0]
    fog_regions = _find_contiguous_regions(labels, 1)
    for start, end in fog_regions:
        ax.axvspan(time_sec[start], time_sec[min(end, len(time_sec)-1)],
                   alpha=0.35, color="#E74C3C", linewidth=0)
    ax.set_ylabel("FoG")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.3)
    ax.set_title(f"Subject S{subject_id:02d} — Biomarker Alignment Analysis", fontweight="bold")
    ax.legend([Patch(facecolor="#E74C3C", alpha=0.35)], ["Ground Truth FoG"],
              loc="upper right", framealpha=0.9)

    # Panel 2: Dynamics residual
    ax = axes[1]
    ax.plot(time_sec, residuals, color="#2C3E50", linewidth=0.6, alpha=0.8)
    tau_r = ckpt.get("tau_r", np.percentile(residuals, 95))
    if np.isfinite(tau_r):
        ax.axhline(y=tau_r, color="#E74C3C", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"$\\tau_r = {tau_r:.4f}$")
    # Shade FoG regions
    for start, end in fog_regions:
        ax.axvspan(time_sec[start], time_sec[min(end, len(time_sec)-1)],
                   alpha=0.12, color="#E74C3C", linewidth=0)
    ax.set_ylabel("Residual $r_{max}$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # Panel 3: Phase advance
    ax = axes[2]
    ax.plot(time_sec, phases, color="#27AE60", linewidth=0.6, alpha=0.8)
    tau_phi = ckpt.get("tau_phi", np.percentile(phases, 5))
    if np.isfinite(tau_phi):
        ax.axhline(y=tau_phi, color="#E74C3C", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"$\\tau_\\phi = {tau_phi:.4f}$")
    for start, end in fog_regions:
        ax.axvspan(time_sec[start], time_sec[min(end, len(time_sec)-1)],
                   alpha=0.12, color="#E74C3C", linewidth=0)
    ax.set_ylabel("Phase $\\Delta\\Phi$ (rad)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    save_path = cfg.results_dir / "figures" / f"biomarker_alignment_S{subject_id:02d}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_limit_cycle_portrait(fold=1, n_samples=30):
    """
    Publication figure: Latent phase-plane portrait.

    Shows normal gait as coherent limit cycles and FoG windows
    as disrupted/collapsed trajectories.
    """
    model, windows, labels, ckpt, device = load_model_and_data(fold)

    normal_idx = np.where(labels == 0)[0]
    fog_idx = np.where(labels == 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Normal trajectories
    ax = axes[0]
    n = min(n_samples, len(normal_idx))
    chosen = np.random.choice(normal_idx, n, replace=False)
    for idx in chosen:
        w = windows[idx:idx + 1]
        with torch.no_grad():
            out = model.forward(w)
            z = out["z_encoded"][:, 0, :].cpu().numpy()
            
        ax.plot(z[:, 0], z[:, 1], alpha=0.5, linewidth=1.5, color="#3498DB")
        
        ax.scatter(z[0, 0], z[0, 1], s=8, color="#2ECC71", zorder=5)

    # Draw target limit cycle
    if hasattr(model.ode_func, 'mu'):
        r = np.sqrt(model.ode_func.mu)
        cx = model.ode_func.cx.item() if hasattr(model.ode_func, 'cx') else 0
        cy = model.ode_func.cy.item() if hasattr(model.ode_func, 'cy') else 0
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 'k--', alpha=0.3,
                linewidth=1.0, label=f"Target $r={r:.2f}$")
        ax.legend(fontsize=7, loc="lower left")

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Normal Gait", fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    # FoG trajectories
    ax = axes[1]
    if len(fog_idx) > 0:
        n = min(n_samples, len(fog_idx))
        chosen = np.random.choice(fog_idx, n, replace=False)
        for idx in chosen:
            w = windows[idx:idx + 1]
            with torch.no_grad():
                out = model.forward(w)
                z = out["z_encoded"][:, 0, :].cpu().numpy()
            ax.plot(z[:, 0], z[:, 1], alpha=0.5, linewidth=1.5, color="#E74C3C")
            
            ax.scatter(z[0, 0], z[0, 1], s=8, color="#2ECC71", zorder=5)

        if hasattr(model.ode_func, 'mu'):
            cx = model.ode_func.cx.item() if hasattr(model.ode_func, 'cx') else 0
            cy = model.ode_func.cy.item() if hasattr(model.ode_func, 'cy') else 0
            ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 'k--', alpha=0.3, linewidth=1.0)
    else:
        ax.text(0.5, 0.5, "No FoG in\ntest fold", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Freezing of Gait", fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    
    # Tighten axes to focus on the unit circle
    for ax in axes:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

    plt.suptitle("Latent Phase-Plane Portrait", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = cfg.results_dir / "figures" / f"limit_cycle_portrait_fold{fold:02d}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def compute_biomarker_correlation(fold=1):
    """
    Quantify how well biomarkers align with FoG.

    Returns:
      - residual_auc: AUC of residual_max for FoG detection
      - phase_auc: AUC of (-phase_advance) for FoG detection
      - onset_alignment: fraction of FoG episodes where residual
        exceeds threshold within 2 windows of annotated onset
    """
    from sklearn.metrics import roc_auc_score

    model, windows, labels, ckpt, device = load_model_and_data(fold)

    all_r, all_p = [], []
    with torch.no_grad():
        for i in range(0, len(windows), 128):
            batch = windows[i:i + 128]
            scores = model.compute_anomaly_scores(batch)
            all_r.append(scores["residual_max"].cpu().numpy())
            all_p.append(scores["phase_advance"].cpu().numpy())

    residuals = np.concatenate(all_r)
    phases = np.concatenate(all_p)

    # Check if there are FoG samples
    if labels.sum() == 0 or labels.sum() == len(labels):
        print(f"  Fold {fold}: Cannot compute correlation (no FoG or all FoG)")
        return {"residual_auc": float("nan"), "phase_auc": float("nan"), "onset_alignment": float("nan")}

    r_auc = roc_auc_score(labels, residuals)
    p_auc = roc_auc_score(labels, -phases)

    # Onset alignment: for each FoG episode onset, check if residual
    # spikes within ±2 windows
    tau_r = ckpt.get("tau_r", np.percentile(residuals, 95))
    fog_regions = _find_contiguous_regions(labels, 1)
    aligned = 0
    for start, end in fog_regions:
        window = slice(max(0, start - 2), min(len(residuals), start + 3))
        if np.any(residuals[window] >= tau_r):
            aligned += 1

    onset_align = aligned / max(len(fog_regions), 1)

    print(f"  Fold {fold}: residual_AUC={r_auc:.3f}, phase_AUC={p_auc:.3f}, "
          f"onset_alignment={onset_align:.1%}")

    return {"residual_auc": r_auc, "phase_auc": p_auc, "onset_alignment": onset_align}


def _find_contiguous_regions(arr, value):
    """Find start/end indices of contiguous regions where arr == value."""
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(arr):
        if v == value and not in_region:
            in_region = True
            start = i
        elif v != value and in_region:
            in_region = False
            regions.append((start, i))
    if in_region:
        regions.append((start, len(arr)))
    return regions


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    args = p.parse_args()

    print(f"\n=== Biomarker Analysis for Fold {args.fold} ===\n")
    plot_biomarker_alignment(args.fold)
    plot_limit_cycle_portrait(args.fold)
    compute_biomarker_correlation(args.fold)

"""
visualization/latent_plots.py — Latent trajectory visualization for Approach 2 (GRU-PINN).

Shows:
  1. Latent trajectory (z1 vs z2) — normal (circle) vs FoG (collasped/chaotic)
  2. Latent radius over time
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from data.dataset import GaitDataset, normalise_fold, apply_normalisation
from models.gru_pinn import GaitGRUPINN

COLORS = {"normal": "#2196F3", "fog": "#F44336"}

def plot_latent_trajectories(model, dataset, device, subject_id=1):
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    mask = dataset.subject_ids == subject_id
    windows = dataset.windows[mask]
    labels = dataset.labels[mask]

    # Find one normal and one FoG window
    normal_idx = (labels == 0).nonzero().flatten()
    fog_idx = (labels == 1).nonzero().flatten()

    if len(normal_idx) == 0 or len(fog_idx) == 0:
        return

    # Pick middle of respective samples
    n_w = windows[normal_idx[len(normal_idx)//2]].unsqueeze(0).to(device)
    f_w = windows[fog_idx[len(fog_idx)//2]].unsqueeze(0).to(device)

    with torch.no_grad():
        n_z = model.encoder(n_w).cpu().numpy()[0]  # (T, 2)
        f_z = model.encoder(f_w).cpu().numpy()[0]  # (T, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Latent Space Analysis — Subject {subject_id}")

    # Normal trajectory
    ax1.plot(n_z[:, 0], n_z[:, 1], color=COLORS["normal"], label="Normal (Limit Cycle)")
    ax1.scatter(n_z[0, 0], n_z[0, 1], marker="o", color="green", label="Start")
    ax1.set_title("Normal Walking (Stable Orbit)")
    ax1.set_xlabel("z1")
    ax1.set_ylabel("z2")
    ax1.axis("equal")
    ax1.legend()

    # FoG trajectory
    ax2.plot(f_z[:, 0], f_z[:, 1], color=COLORS["fog"], label="FoG (Collapsed/Chaotic)")
    ax2.scatter(f_z[0, 0], f_z[0, 1], marker="o", color="green")
    ax2.set_title("Freezing of Gait (Dynamic Breakdown)")
    ax2.set_xlabel("z1")
    ax2.set_ylabel("z2")
    ax2.axis("equal")
    ax2.legend()

    plt.tight_layout()
    path = fig_dir / "latent_trajectories.png"
    fig.savefig(path)
    print(f"Saved: {path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GaitDataset()
    # Mock model for checking script (actual trainer will call this after training)
    model = GaitGRUPINN(in_dim=9).to(device)
    plot_latent_trajectories(model, dataset, device)

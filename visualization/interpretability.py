import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from models.pinn import GaitPINN
from data.dataset import GaitDataset, apply_normalisation, normalise_fold
from utils import get_device, load_checkpoint

def plot_interpretability(fold=1, subject_id=1):
    device = get_device()
    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return

    print(f"Loading model from {ckpt_path}...")
    ckpt = load_checkpoint(ckpt_path, device=device)
    
    model = GaitPINN(
        in_dim=cfg.num_channels,
        latent_dim=cfg.latent_dim,
        encoder_hidden=cfg.encoder_hidden,
        ode_hidden=cfg.ode_hidden,
        decoder_out=cfg.decoder_out,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = GaitDataset()
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    
    # Extract data for the specific subject
    mask = dataset.subject_ids == subject_id
    windows = dataset.windows[mask]
    labels = dataset.labels[mask].numpy()
    
    print(f"Processing {len(windows)} windows for Subject {subject_id}...")
    
    norm_windows = apply_normalisation(windows.to(device), mean, std)
    
    all_residuals = []
    all_phases = []
    
    with torch.no_grad():
        for i in range(0, len(norm_windows), 128):
            batch = norm_windows[i : i+128]
            scores = model.compute_anomaly_scores(batch)
            all_residuals.append(scores["residual_max"].cpu().numpy())
            all_phases.append(scores["phase_advance"].cpu().numpy())
            
    residuals = np.concatenate(all_residuals)
    phases = np.concatenate(all_phases)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    time_index = np.arange(len(labels))
    
    # 1. Ground Truth
    axes[0].fill_between(time_index, 0, labels, color='red', alpha=0.3, label='Ground Truth FoG')
    axes[0].set_ylabel("FoG Label")
    axes[0].set_title(f"Subject {subject_id} — Interpretability Analysis")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(loc='upper right')
    
    # 2. Residuals
    axes[1].plot(time_index, residuals, color='blue', label='Dynamics Residual $r_{max}$')
    axes[1].axhline(y=ckpt["tau_r"], color='black', linestyle='--', label='Threshold $\\tau_r$')
    axes[1].set_ylabel("Residual")
    axes[1].legend(loc='upper right')
    
    # 3. Phase Advance
    axes[2].plot(time_index, phases, color='green', label='Phase Advance $\\Delta\\Phi$')
    axes[2].axhline(y=ckpt["tau_phi"], color='black', linestyle='--', label='Threshold $\\tau_\\phi$')
    axes[2].set_ylabel("Phase Advance (rad)")
    axes[2].set_xlabel("Window Index")
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    save_path = cfg.results_dir / "figures" / f"interpretability_S{subject_id:02d}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Interpretability plot saved to {save_path}")

if __name__ == "__main__":
    plot_interpretability(fold=1, subject_id=1)

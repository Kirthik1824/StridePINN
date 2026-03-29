import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from models.pinn import GaitPINN
from data.dataset import GaitDataset, apply_normalisation
from utils import get_device

def diagnose_latent(fold=1, n_samples=30, latent_dim=2, ode_mode="mlp"):
    device = get_device()
    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return

    print(f"Loading fold {fold} model ({latent_dim}D, {ode_mode})...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = GaitPINN(
        in_dim=cfg.num_channels,
        latent_dim=latent_dim,
        encoder_hidden=cfg.encoder_hidden,
        ode_hidden=cfg.ode_hidden,
        decoder_out=cfg.decoder_out,
        ode_mode=ode_mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = GaitDataset()
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    
    # Extract normal windows for the fold subject
    mask = (dataset.subject_ids == fold) & (dataset.labels == 0)
    windows = dataset.windows[mask]
    
    if len(windows) == 0:
        print("No normal windows found for this subject.")
        return

    print(f"Sampling {n_samples} random normal windows...")
    indices = np.random.choice(len(windows), min(n_samples, len(windows)), replace=False)
    sample_windows = apply_normalisation(windows[indices].to(device), mean, std)

    plt.figure(figsize=(8, 8))
    for i in range(len(sample_windows)):
        with torch.no_grad():
            out = model.forward(sample_windows[i:i+1])
            z = out["z_traj"][:, 0, :].cpu().numpy()
            plt.plot(z[:, 0], z[:, 1], alpha=0.3, color='blue')
            plt.scatter(z[0, 0], z[0, 1], s=10, color='green') # Start point

    plt.title(f"Latent Trajectories (Normal Gait) - Fold {fold}")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    save_path = cfg.results_dir / "figures" / f"latent_diagnosis_fold_{fold:02d}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Diagnostic plot saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--ode_mode", type=str, default="mlp")
    args = parser.parse_args()
    diagnose_latent(args.fold, latent_dim=args.latent_dim, ode_mode=args.ode_mode)

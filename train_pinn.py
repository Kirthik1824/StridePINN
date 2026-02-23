"""
train_pinn.py — LOSO training loop for the Physics-Informed Neural Network.

The PINN is trained on normal-gait windows only (anomaly-detection paradigm).
FoG detection uses dynamics residual and phase stagnation at inference time.

Features:
  - 4-term physics-informed loss (data + periodicity + phase + smoothness)
  - Adam with cosine annealing and gradient clipping
  - Per-epoch loss breakdown logging
  - ODE stability monitoring
  - Anomaly-score threshold calibration on validation fold

Usage:
  python train_pinn.py                      # Full LOSO training
  python train_pinn.py --folds 2 --epochs 5 # Quick test run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import cfg
from utils import (
    seed_everything,
    get_device,
    setup_logger,
    save_checkpoint,
    compute_metrics,
    find_optimal_threshold,
    compute_detection_latency,
)
from data.dataset import (
    GaitDataset,
    get_loso_splits,
    normalise_fold,
    apply_normalisation,
    make_dataloader,
)
from models.pinn import GaitPINN


def parse_args():
    p = argparse.ArgumentParser(description="Train PINN gait model")
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--folds", type=int, default=cfg.num_subjects)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--latent_dim", type=int, default=cfg.latent_dim)
    return p.parse_args()


def train_one_fold(
    fold_info: dict,
    dataset: GaitDataset,
    args,
    device: torch.device,
    logger,
) -> dict:
    """Train PINN on a single LOSO fold."""
    fold = fold_info["fold"]
    logger.info(f"{'='*60}")
    logger.info(f"FOLD {fold} — test={fold_info['test_subject']}, val={fold_info['val_subject']}")
    logger.info(f"{'='*60}")

    # --- Normalise ---
    mean, std = normalise_fold(dataset, fold_info["train_idx"])

    # --- Extract training data (NORMAL gait only) ---
    train_idx = fold_info["train_idx"]
    train_mask_normal = dataset.labels[train_idx] == 0
    normal_idx = train_idx[train_mask_normal.numpy()]

    train_w = apply_normalisation(dataset.windows[normal_idx], mean, std)
    train_a = apply_normalisation(dataset.ankle[normal_idx], mean[:3], std[:3])

    # Validation: both normal and FoG for threshold calibration
    val_w = apply_normalisation(dataset.windows[fold_info["val_idx"]], mean, std)
    val_a = apply_normalisation(dataset.ankle[fold_info["val_idx"]], mean[:3], std[:3])
    val_l = dataset.labels[fold_info["val_idx"]]

    # Test
    test_w = apply_normalisation(dataset.windows[fold_info["test_idx"]], mean, std)
    test_a = apply_normalisation(dataset.ankle[fold_info["test_idx"]], mean[:3], std[:3])
    test_l = dataset.labels[fold_info["test_idx"]]

    logger.info(f"  Train (normal only): {len(train_w)} windows")
    logger.info(f"  Val: {len(val_w)} windows ({(val_l==1).sum()} FoG)")
    logger.info(f"  Test: {len(test_w)} windows ({(test_l==1).sum()} FoG)")

    # --- DataLoaders ---
    train_labels_dummy = torch.zeros(len(train_w), dtype=torch.long)
    train_loader = make_dataloader(train_w, train_a, train_labels_dummy, args.batch_size, shuffle=True)

    # --- Model ---
    model = GaitPINN(
        in_dim=cfg.num_channels,
        latent_dim=args.latent_dim,
        encoder_hidden=cfg.encoder_hidden,
        ode_hidden=cfg.ode_hidden,
        decoder_out=cfg.decoder_out,
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
        ode_step_size=cfg.ode_step_size,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "data": 0, "cyc": 0, "phi": 0, "smooth": 0}
        n_batches = 0

        for batch in train_loader:
            windows, ankle, _ = batch
            windows = windows.to(device)
            ankle = ankle.to(device)

            try:
                losses = model.compute_loss(
                    windows, ankle,
                    lambda_cyc=cfg.lambda_cyc,
                    lambda_phi=cfg.lambda_phi,
                    lambda_smooth=cfg.lambda_smooth,
                )

                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                for k in epoch_losses:
                    epoch_losses[k] += losses[k].item()
                n_batches += 1

            except RuntimeError as e:
                if "underflow" in str(e).lower() or "overflow" in str(e).lower():
                    logger.warning(f"  ODE solver error at epoch {epoch}: {e}")
                    continue
                raise

        scheduler.step()

        if n_batches == 0:
            logger.warning(f"  Epoch {epoch}: no valid batches, skipping")
            continue

        avg_loss = epoch_losses["total"] / n_batches

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d} — "
                f"total={avg_loss:.4f}, "
                f"data={epoch_losses['data']/n_batches:.4f}, "
                f"cyc={epoch_losses['cyc']/n_batches:.4f}, "
                f"phi={epoch_losses['phi']/n_batches:.5f}, "
                f"smooth={epoch_losses['smooth']/n_batches:.5f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- Load best model ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # --- Anomaly scoring on validation set ---
    logger.info("  Computing anomaly scores on validation set...")
    val_scores = compute_anomaly_scores_dataset(model, val_w, device, args.batch_size)

    val_true_np = val_l.numpy()
    residual_max = val_scores["residual_max"]
    phase_advance = val_scores["phase_advance"]

    # Calibrate thresholds via Youden index
    # Residual: higher = more abnormal
    tau_r = find_optimal_threshold(val_true_np, residual_max)
    # Phase advance: lower = more abnormal → invert for threshold finding
    tau_phi_inv = find_optimal_threshold(val_true_np, -phase_advance)
    tau_phi = -tau_phi_inv  # actual threshold: flag if phase_advance < tau_phi

    logger.info(f"  Calibrated thresholds: τ_r={tau_r:.4f}, τ_φ={tau_phi:.4f}")

    # --- Test evaluation ---
    logger.info("  Evaluating on test set...")
    test_scores = compute_anomaly_scores_dataset(model, test_w, device, args.batch_size)

    test_true_np = test_l.numpy()

    # OR-rule: flag if residual > τ_r OR phase_advance < τ_φ
    y_pred_r = (test_scores["residual_max"] >= tau_r).astype(int)
    y_pred_phi = (test_scores["phase_advance"] <= tau_phi).astype(int)
    y_pred = np.maximum(y_pred_r, y_pred_phi)  # logical OR

    # Use residual_max as continuous score for AUC
    test_metrics = compute_metrics(test_true_np, test_scores["residual_max"], threshold=tau_r)

    # Detection latency
    latency = compute_detection_latency(test_true_np, y_pred, window_stride_sec=0.8)

    logger.info(f"  TEST — AUC={test_metrics['auc']:.3f}, Se={test_metrics['sensitivity']:.3f}, "
                f"Sp={test_metrics['specificity']:.3f}, F1={test_metrics['f1']:.3f}, "
                f"latency={latency['median']:.2f}s")

    # Save checkpoint
    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    save_checkpoint({
        "model_state": best_state or model.state_dict(),
        "fold": fold,
        "test_metrics": test_metrics,
        "tau_r": tau_r,
        "tau_phi": tau_phi,
        "norm_mean": mean,
        "norm_std": std,
        "best_loss": best_loss,
    }, ckpt_path)

    return {
        "fold": fold,
        "test_metrics": test_metrics,
        "tau_r": tau_r,
        "tau_phi": tau_phi,
        "latency": latency,
        "best_loss": best_loss,
    }


def compute_anomaly_scores_dataset(
    model: GaitPINN,
    windows: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Compute anomaly scores for all windows in batches."""
    model.eval()
    all_residual_max = []
    all_phase_advance = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size].to(device)
            scores = model.compute_anomaly_scores(batch)
            all_residual_max.append(scores["residual_max"].cpu().numpy())
            all_phase_advance.append(scores["phase_advance"].cpu().numpy())

    return {
        "residual_max": np.concatenate(all_residual_max),
        "phase_advance": np.concatenate(all_phase_advance),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_pinn", log_file=log_dir / "train_pinn.log")

    logger.info(f"Device: {device}")
    logger.info(f"Latent dim: {args.latent_dim}, Epochs: {args.epochs}, Folds: {args.folds}")
    logger.info(f"Loss weights: λ_cyc={cfg.lambda_cyc}, λ_φ={cfg.lambda_phi}, λ_s={cfg.lambda_smooth}")

    # Load dataset
    dataset = GaitDataset()

    # Run LOSO
    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds:
            break
        result = train_one_fold(fold_info, dataset, args, device, logger)
        all_results.append(result)

    # Aggregate metrics
    metrics_keys = ["auc", "sensitivity", "specificity", "f1"]
    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE RESULTS (PINN, {len(all_results)} folds)")
    logger.info(f"{'='*60}")

    for key in metrics_keys:
        values = [r["test_metrics"][key] for r in all_results]
        logger.info(f"  {key:15s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    latencies = [r["latency"]["median"] for r in all_results if not np.isnan(r["latency"]["median"])]
    if latencies:
        logger.info(f"  {'latency (s)':15s}: {np.mean(latencies):.2f} ± {np.std(latencies):.2f}")

    # Save aggregate results
    results_path = cfg.results_dir / "pinn_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

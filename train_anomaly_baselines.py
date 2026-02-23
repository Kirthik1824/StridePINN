"""
train_anomaly_baselines.py — LOSO training for anomaly detection baselines.

Models:
  - conv_ae:   Convolutional Autoencoder (deep, normal-only)
  - ocsvm:     One-Class SVM (classical, normal-only)

All models train on normal gait only, same protocol as the PINN.
FoG is detected as anomaly (high reconstruction error / negative decision).

Usage:
  python train_anomaly_baselines.py --model conv_ae
  python train_anomaly_baselines.py --model ocsvm
  python train_anomaly_baselines.py --model conv_ae --folds 2 --epochs 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.svm import OneClassSVM
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
from models.conv_ae import FoGConvAE


def parse_args():
    p = argparse.ArgumentParser(description="Train anomaly detection baselines")
    p.add_argument("--model", type=str, default="conv_ae", choices=["conv_ae", "ocsvm"])
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--folds", type=int, default=cfg.num_subjects,
                    help="Number of LOSO folds (default: all 10)")
    p.add_argument("--seed", type=int, default=cfg.seed)
    return p.parse_args()


# -----------------------------------------------------------------
#  Convolutional Autoencoder Training
# -----------------------------------------------------------------
def train_conv_ae_fold(fold_info, dataset, args, device, logger):
    """Train Conv AE on normal gait only, evaluate on all test data."""
    fold = fold_info["fold"]
    logger.info(f"{'='*60}")
    logger.info(f"FOLD {fold} — test={fold_info['test_subject']}, val={fold_info['val_subject']}")
    logger.info(f"{'='*60}")

    # Normalise
    mean, std = normalise_fold(dataset, fold_info["train_idx"])

    # Extract fold data — TRAIN ON NORMAL ONLY
    train_all_w = apply_normalisation(dataset.windows[fold_info["train_idx"]], mean, std)
    train_all_l = dataset.labels[fold_info["train_idx"]]
    train_normal_mask = train_all_l == 0
    train_w = train_all_w[train_normal_mask]
    train_a = dataset.ankle[fold_info["train_idx"]][train_normal_mask]
    train_l = train_all_l[train_normal_mask]

    val_w = apply_normalisation(dataset.windows[fold_info["val_idx"]], mean, std)
    val_l = dataset.labels[fold_info["val_idx"]]

    test_w = apply_normalisation(dataset.windows[fold_info["test_idx"]], mean, std)
    test_l = dataset.labels[fold_info["test_idx"]]

    logger.info(f"  Train (normal only): {len(train_w)} windows")

    # DataLoaders
    train_loader = make_dataloader(train_w, train_a, train_l, args.batch_size, shuffle=True)

    # Model
    model = FoGConvAE(
        in_channels=cfg.num_channels,
        latent_dim=64,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Training loop
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            windows, _, _ = batch
            windows = windows.to(device)

            x_hat = model(windows)
            loss = criterion(x_hat, windows)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d} — loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Compute anomaly scores on val + test
    def get_scores(windows):
        scores = []
        with torch.no_grad():
            for i in range(0, len(windows), args.batch_size):
                batch = windows[i:i + args.batch_size].to(device)
                s = model.anomaly_score(batch).cpu().numpy()
                scores.append(s)
        return np.concatenate(scores)

    val_scores = get_scores(val_w)
    test_scores = get_scores(test_w)

    val_true = val_l.numpy()
    test_true = test_l.numpy()

    # Calibrate threshold on val set (higher score → more anomalous → FoG)
    opt_thresh = find_optimal_threshold(val_true, val_scores)
    test_metrics = compute_metrics(test_true, test_scores, threshold=opt_thresh)

    # Detection latency
    test_pred = (test_scores >= opt_thresh).astype(int)
    latency = compute_detection_latency(test_true, test_pred, window_stride_sec=0.8)

    logger.info(f"  TEST — AUC={test_metrics['auc']:.3f}, Se={test_metrics['sensitivity']:.3f}, "
                f"Sp={test_metrics['specificity']:.3f}, F1={test_metrics['f1']:.3f}, "
                f"latency={latency['median']:.2f}s")

    # Save checkpoint
    ckpt_path = cfg.checkpoint_dir / "conv_ae" / f"fold_{fold:02d}.pt"
    save_checkpoint({
        "model_state": best_state or model.state_dict(),
        "fold": fold,
        "test_metrics": test_metrics,
        "threshold": opt_thresh,
        "norm_mean": mean,
        "norm_std": std,
    }, ckpt_path)

    return {
        "fold": fold,
        "test_metrics": test_metrics,
        "threshold": float(opt_thresh),
        "latency": latency,
        "best_loss": float(best_loss),
    }


# -----------------------------------------------------------------
#  One-Class SVM Training
# -----------------------------------------------------------------
def train_ocsvm_fold(fold_info, dataset, args, logger):
    """Train OC-SVM on normal gait only, evaluate on all test data."""
    fold = fold_info["fold"]
    logger.info(f"{'='*60}")
    logger.info(f"FOLD {fold} — test={fold_info['test_subject']}, val={fold_info['val_subject']}")
    logger.info(f"{'='*60}")

    # Normalise
    mean, std = normalise_fold(dataset, fold_info["train_idx"])

    # Extract fold data — TRAIN ON NORMAL ONLY
    train_all_w = apply_normalisation(dataset.windows[fold_info["train_idx"]], mean, std)
    train_all_l = dataset.labels[fold_info["train_idx"]]
    train_normal_mask = train_all_l == 0
    train_w = train_all_w[train_normal_mask]

    val_w = apply_normalisation(dataset.windows[fold_info["val_idx"]], mean, std)
    val_l = dataset.labels[fold_info["val_idx"]]

    test_w = apply_normalisation(dataset.windows[fold_info["test_idx"]], mean, std)
    test_l = dataset.labels[fold_info["test_idx"]]

    # Flatten windows: (N, 128, 9) → (N, 1152)
    train_flat = train_w.reshape(len(train_w), -1).numpy()
    val_flat = val_w.reshape(len(val_w), -1).numpy()
    test_flat = test_w.reshape(len(test_w), -1).numpy()

    logger.info(f"  Train (normal only): {len(train_flat)} windows, dim={train_flat.shape[1]}")

    # Subsample if too many training windows (OC-SVM is O(n²))
    max_train = 5000
    if len(train_flat) > max_train:
        idx = np.random.choice(len(train_flat), max_train, replace=False)
        train_flat = train_flat[idx]
        logger.info(f"  Subsampled to {max_train} for OC-SVM scalability")

    # Fit OC-SVM
    logger.info("  Fitting One-Class SVM (rbf kernel)...")
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    svm.fit(train_flat)
    logger.info("  OC-SVM fitted.")

    # Score: decision_function returns negative for outliers
    # We negate so higher = more anomalous (consistent with other models)
    val_scores = -svm.decision_function(val_flat)
    test_scores = -svm.decision_function(test_flat)

    val_true = val_l.numpy()
    test_true = test_l.numpy()

    # Calibrate threshold
    opt_thresh = find_optimal_threshold(val_true, val_scores)
    test_metrics = compute_metrics(test_true, test_scores, threshold=opt_thresh)

    # Detection latency
    test_pred = (test_scores >= opt_thresh).astype(int)
    latency = compute_detection_latency(test_true, test_pred, window_stride_sec=0.8)

    logger.info(f"  TEST — AUC={test_metrics['auc']:.3f}, Se={test_metrics['sensitivity']:.3f}, "
                f"Sp={test_metrics['specificity']:.3f}, F1={test_metrics['f1']:.3f}, "
                f"latency={latency['median']:.2f}s")

    return {
        "fold": fold,
        "test_metrics": test_metrics,
        "threshold": float(opt_thresh),
        "latency": latency,
    }


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"train_{args.model}",
        log_file=log_dir / f"train_{args.model}.log",
    )

    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}, Epochs: {args.epochs}, Folds: {args.folds}")

    # Load dataset
    dataset = GaitDataset()

    # Run LOSO
    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds:
            break

        if args.model == "conv_ae":
            result = train_conv_ae_fold(fold_info, dataset, args, device, logger)
        elif args.model == "ocsvm":
            result = train_ocsvm_fold(fold_info, dataset, args, logger)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        all_results.append(result)

    # Aggregate metrics
    metrics_keys = ["auc", "sensitivity", "specificity", "f1"]
    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE RESULTS ({args.model.upper()}, {len(all_results)} folds)")
    logger.info(f"{'='*60}")

    for key in metrics_keys:
        values = [r["test_metrics"][key] for r in all_results]
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            logger.info(f"  {key:15s}: {np.mean(valid):.3f} ± {np.std(valid):.3f} ({len(valid)} valid folds)")
        else:
            logger.info(f"  {key:15s}: N/A (all NaN)")

    latencies = [r["latency"]["median"] for r in all_results
                 if r.get("latency") and not np.isnan(r["latency"]["median"])]
    if latencies:
        logger.info(f"  {'latency (s)':15s}: {np.mean(latencies):.2f} ± {np.std(latencies):.2f}")

    # Save results
    results_path = cfg.results_dir / f"{args.model}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

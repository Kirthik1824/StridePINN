"""
train_baselines.py — LOSO training loop for 1D-CNN and CNN-LSTM baselines.

Features:
  - Inverse-frequency weighted BCE loss
  - FoG oversampling with ±4-sample temporal jitter
  - Adam + cosine-annealing LR schedule
  - Early stopping on validation F1
  - Per-fold checkpoint saving and metric logging

Usage:
  python train_baselines.py --model cnn          # Train 1D-CNN
  python train_baselines.py --model cnn_lstm     # Train CNN-LSTM
  python train_baselines.py --model cnn --folds 2 --epochs 5  # Quick test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    augment_fog_windows,
    make_dataloader,
)
from models.cnn import FoGCNN1D
from models.cnn_lstm import FoGCNNLSTM


def parse_args():
    p = argparse.ArgumentParser(description="Train FoG baseline classifiers")
    p.add_argument("--model", type=str, default="cnn", choices=["cnn", "cnn_lstm"])
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--folds", type=int, default=cfg.num_subjects,
                    help="Number of LOSO folds to run (default: all 10)")
    p.add_argument("--seed", type=int, default=cfg.seed)
    return p.parse_args()


def build_model(model_type: str, device: torch.device) -> nn.Module:
    if model_type == "cnn":
        model = FoGCNN1D(
            in_channels=cfg.num_channels,
            conv1_out=cfg.cnn_conv1_out,
            conv2_out=cfg.cnn_conv2_out,
            kernel_size=cfg.cnn_kernel_size,
            dropout=cfg.cnn_dropout,
        )
    elif model_type == "cnn_lstm":
        model = FoGCNNLSTM(
            in_channels=cfg.num_channels,
            conv1_out=cfg.lstm_conv1_out,
            conv2_out=cfg.lstm_conv2_out,
            kernel_size=cfg.lstm_kernel_size,
            lstm_hidden=cfg.lstm_hidden,
            dropout=cfg.lstm_dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def train_one_fold(
    fold_info: dict,
    dataset: GaitDataset,
    model_type: str,
    args,
    device: torch.device,
    logger,
) -> dict:
    """Train and evaluate on a single LOSO fold."""
    fold = fold_info["fold"]
    logger.info(f"{'='*60}")
    logger.info(f"FOLD {fold} — test={fold_info['test_subject']}, val={fold_info['val_subject']}")
    logger.info(f"{'='*60}")

    # --- Normalise ---
    mean, std = normalise_fold(dataset, fold_info["train_idx"])

    # Extract fold data
    train_w = apply_normalisation(dataset.windows[fold_info["train_idx"]], mean, std)
    train_a = dataset.ankle[fold_info["train_idx"]]
    train_l = dataset.labels[fold_info["train_idx"]]

    val_w = apply_normalisation(dataset.windows[fold_info["val_idx"]], mean, std)
    val_l = dataset.labels[fold_info["val_idx"]]

    test_w = apply_normalisation(dataset.windows[fold_info["test_idx"]], mean, std)
    test_l = dataset.labels[fold_info["test_idx"]]

    # --- Augment training FoG windows ---
    train_w, train_a, train_l = augment_fog_windows(
        train_w, train_a, train_l,
        oversample_factor=cfg.fog_oversample_factor,
        jitter=cfg.jitter_samples,
    )

    # --- Inverse-frequency loss weighting ---
    n_normal = (train_l == 0).sum().item()
    n_fog = (train_l == 1).sum().item()
    pos_weight = torch.tensor([n_normal / max(n_fog, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.info(f"  Train: {len(train_l)} windows (normal={n_normal}, FoG={n_fog}), "
                f"pos_weight={pos_weight.item():.2f}")

    # --- DataLoaders ---
    train_loader = make_dataloader(train_w, train_a, train_l, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_w, val_w[:, :, :3], val_l, args.batch_size, shuffle=False)
    test_loader = make_dataloader(test_w, test_w[:, :, :3], test_l, args.batch_size, shuffle=False)

    # --- Model, Optimizer, Scheduler ---
    model = build_model(model_type, device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            windows, _, labels = batch
            windows = windows.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(windows)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                windows, _, labels = batch
                probs = model.predict_proba(windows.to(device)).cpu().numpy()
                val_probs.append(probs.squeeze())
                val_true.append(labels.numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_metrics = compute_metrics(val_true, val_probs)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d} — loss={epoch_loss/n_batches:.4f}, "
                f"val_AUC={val_metrics['auc']:.3f}, val_F1={val_metrics['f1']:.3f}"
            )

        # Early stopping on val F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # --- Test evaluation ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_probs, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            windows, _, labels = batch
            probs = model.predict_proba(windows.to(device)).cpu().numpy()
            test_probs.append(probs.squeeze())
            test_true.append(labels.numpy())

    test_probs = np.concatenate(test_probs)
    test_true = np.concatenate(test_true)

    # Optimal threshold from validation
    opt_thresh = find_optimal_threshold(val_true, val_probs)
    test_metrics = compute_metrics(test_true, test_probs, threshold=opt_thresh)

    # Detection latency
    test_pred = (test_probs >= opt_thresh).astype(int)
    latency = compute_detection_latency(test_true, test_pred, window_stride_sec=0.8)

    logger.info(f"  TEST — AUC={test_metrics['auc']:.3f}, Se={test_metrics['sensitivity']:.3f}, "
                f"Sp={test_metrics['specificity']:.3f}, F1={test_metrics['f1']:.3f}, "
                f"latency={latency['median']:.2f}s")

    # Save checkpoint
    ckpt_path = cfg.checkpoint_dir / model_type / f"fold_{fold:02d}.pt"
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
        "threshold": opt_thresh,
        "latency": latency,
    }


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
        result = train_one_fold(fold_info, dataset, args.model, args, device, logger)
        all_results.append(result)

    # Aggregate metrics
    metrics_keys = ["auc", "sensitivity", "specificity", "f1"]
    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE RESULTS ({args.model.upper()}, {len(all_results)} folds)")
    logger.info(f"{'='*60}")

    for key in metrics_keys:
        values = [r["test_metrics"][key] for r in all_results]
        logger.info(f"  {key:15s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    latencies = [r["latency"]["median"] for r in all_results if not np.isnan(r["latency"]["median"])]
    if latencies:
        logger.info(f"  {'latency (s)':15s}: {np.mean(latencies):.2f} ± {np.std(latencies):.2f}")

    # Save aggregate results
    results_path = cfg.results_dir / f"{args.model}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

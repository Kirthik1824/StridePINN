"""
training/prediction_trainer.py — Training loop for Approach 1: FoG Prediction.

Trains the CNN-LSTM+Physics+EWS model to predict whether FoG will occur
in the next k seconds. Uses shifted labelling: each window's target is
whether FoG exists in the future horizon.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from data.dataset import (
    GaitDataset, normalise_fold, apply_normalisation,
    augment_fog_windows, create_weighted_sampler, make_dataloader,
)
from training.trainer_utils import prepare_fold_data
from models.cnn_lstm_prediction import FoGCNNLSTMPrediction
from features import extract_features_batch, get_feature_names
from features_ews import extract_ews_features_batch, get_ews_feature_names
from utils import compute_metrics, find_optimal_threshold, compute_lead_time


def create_prediction_labels(
    labels: np.ndarray,
    horizon_windows: int,
) -> np.ndarray:
    """
    Create shifted labels for the prediction task.

    For each window i, the prediction label is 1 if ANY of the windows
    in [i+1, i+horizon_windows] has label 1 (FoG).

    Args:
        labels: (N,) binary labels
        horizon_windows: number of future windows to look ahead

    Returns:
        pred_labels: (N,) prediction targets
    """
    n = len(labels)
    pred_labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        future_start = i + 1
        future_end = min(i + 1 + horizon_windows, n)
        if future_start < n and np.any(labels[future_start:future_end] == 1):
            pred_labels[i] = 1

    return pred_labels

def _extract_physics_features(windows_np: np.ndarray) -> torch.Tensor:
    """Extract physics features and return as tensor."""
    feats = extract_features_batch(windows_np, fs=cfg.target_fs, tau=5, channel=0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.tensor(feats, dtype=torch.float32)


def _extract_ews_features(windows_np: np.ndarray) -> torch.Tensor:
    """Extract EWS features and return as tensor."""
    feats = extract_ews_features_batch(
        windows_np, fs=cfg.target_fs, tau=5,
        sub_window=cfg.ews_sub_window, channel=0,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.tensor(feats, dtype=torch.float32)


def get_cached_features(dataset: GaitDataset, logger=None):
    if not hasattr(dataset, "phys_features"):
        if logger: logger.info("  [Cache] Computing physics features for entire dataset...")
        dataset.phys_features = _extract_physics_features(dataset.windows.numpy())
        if logger: logger.info("  [Cache] Computing EWS features for entire dataset...")
        dataset.ews_features = _extract_ews_features(dataset.windows.numpy())
    return dataset.phys_features, dataset.ews_features


def train_prediction_fold(
    fold_info: dict,
    dataset: GaitDataset,
    args,
    device: torch.device,
    logger=None,
) -> dict:
    """
    Train one LOSO fold of the CNN-LSTM+EWS prediction model.

    Args:
        fold_info: dict with fold, train_idx, val_idx, test_idx
        dataset: GaitDataset instance
        args: CLI arguments (epochs, lr, batch_size, horizon, etc.)
        device: torch device
        logger: optional logger

    Returns:
        dict with fold results including test_metrics, latency
    """
    fold = fold_info["fold"]
    horizon_sec = getattr(args, "horizon", cfg.default_prediction_horizon)
    window_stride_sec = cfg.window_stride / cfg.target_fs
    horizon_windows = max(1, int(horizon_sec / window_stride_sec))

    if logger:
        logger.info(f"Fold {fold}: horizon={horizon_sec}s ({horizon_windows} windows)")

    # --- Data preparation ---
    data = prepare_fold_data(dataset, fold_info, device)
    train_w, train_a, train_l = data["train"]
    val_w, val_a, val_l = data["val"]
    test_w, test_a, test_l = data["test"]

    # Create prediction labels (shifted)
    train_pred_l = torch.tensor(
        create_prediction_labels(train_l.numpy(), horizon_windows), dtype=torch.long
    )
    val_pred_l = torch.tensor(
        create_prediction_labels(val_l.numpy(), horizon_windows), dtype=torch.long
    )
    test_pred_l = torch.tensor(
        create_prediction_labels(test_l.numpy(), horizon_windows), dtype=torch.long
    )

    # Extract features (cached globally on dataset)
    all_phys, all_ews = get_cached_features(dataset, logger)
    
    train_phys = all_phys[fold_info["train_idx"]]
    val_phys = all_phys[fold_info["val_idx"]]
    test_phys = all_phys[fold_info["test_idx"]]

    train_ews = all_ews[fold_info["train_idx"]]
    val_ews = all_ews[fold_info["val_idx"]]
    test_ews = all_ews[fold_info["test_idx"]]

    # Augmentation
    train_w_aug, train_a_aug, train_pred_l_aug = augment_fog_windows(
        train_w, train_a, train_pred_l,
        oversample_factor=cfg.fog_oversample_factor,
        jitter=cfg.jitter_samples,
    )

    # Need to replicate physics/ews features for augmented FoG windows
    fog_mask = train_pred_l == 1
    n_aug_copies = cfg.fog_oversample_factor - 1
    extra_phys = train_phys[fog_mask].repeat(n_aug_copies, 1)
    extra_ews = train_ews[fog_mask].repeat(n_aug_copies, 1)
    train_phys_aug = torch.cat([train_phys, extra_phys], dim=0)
    train_ews_aug = torch.cat([train_ews, extra_ews], dim=0)

    # Weighted sampler
    sampler = create_weighted_sampler(train_pred_l_aug)

    # DataLoader (custom collation for extra features)
    train_ds = torch.utils.data.TensorDataset(
        train_w_aug, train_phys_aug, train_ews_aug, train_pred_l_aug
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    val_ds = torch.utils.data.TensorDataset(val_w, val_phys, val_ews, val_pred_l)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    # --- Model ---
    num_phys = train_phys.shape[1]
    num_ews = train_ews.shape[1]

    model = FoGCNNLSTMPrediction(
        in_channels=cfg.num_channels,
        conv1_out=cfg.lstm_conv1_out,
        conv2_out=cfg.lstm_conv2_out,
        kernel_size=cfg.lstm_kernel_size,
        lstm_hidden=cfg.lstm_hidden,
        num_phys_features=num_phys,
        num_ews_features=num_ews,
        dropout=cfg.lstm_dropout,
        use_physics=getattr(args, "use_physics", True),
        use_ews=getattr(args, "use_ews", True),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            windows, phys, ews, labels = [b.to(device) for b in batch]
            logits = model(windows, phys, ews)
            loss = criterion(logits.squeeze(-1), labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_probs = []
        val_labels_list = []
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                windows, phys, ews, labels = [b.to(device) for b in batch]
                logits = model(windows, phys, ews)
                probs = torch.sigmoid(logits).squeeze(-1)
                
                loss = criterion(logits.squeeze(-1), labels.float())
                val_loss += loss.item()
                n_val_batches += 1
                
                val_probs.append(probs.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_labels_arr = np.concatenate(val_labels_list)
        val_metrics = compute_metrics(val_labels_arr, val_probs)
        val_loss /= max(n_val_batches, 1)

        # For zero-FoG validation folds, AUC is NaN. Fallback to loss.
        is_val_auc_nan = np.isnan(val_metrics["auc"])
        
        improved = False
        if not is_val_auc_nan:
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                improved = True
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True

        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 and logger:
            logger.info(
                f"  Fold {fold} Epoch {epoch}: loss={train_loss/max(n_batches,1):.4f}, "
                f"val_AUC={val_metrics['auc']:.3f}"
            )

        if patience_counter >= cfg.early_stop_patience:
            if logger:
                logger.info(f"  Fold {fold}: early stop at epoch {epoch}")
            break

    # --- Test ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_ds = torch.utils.data.TensorDataset(test_w, test_phys, test_ews, test_pred_l)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    test_probs = []
    test_labels_list = []
    with torch.no_grad():
        for batch in test_loader:
            windows, phys, ews, labels = [b.to(device) for b in batch]
            probs = torch.sigmoid(model(windows, phys, ews)).squeeze(-1)
            test_probs.append(probs.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())

    test_probs = np.concatenate(test_probs)
    test_labels_arr = np.concatenate(test_labels_list)

    threshold = find_optimal_threshold(test_labels_arr, test_probs)
    test_metrics = compute_metrics(test_labels_arr, test_probs, threshold=threshold)

    y_pred = (test_probs >= threshold).astype(int)
    
    # Lead time computed against the ORIGINAL labels, not the shifted ones
    latency = compute_lead_time(
        test_l.numpy(), y_pred,
        window_stride_sec=window_stride_sec,
        horizon_windows=horizon_windows,
    )

    if logger:
        logger.info(
            f"Fold {fold} TEST: AUC={test_metrics['auc']:.3f}, "
            f"F1={test_metrics['f1']:.3f}, "
            f"Sens={test_metrics['sensitivity']:.3f}, "
            f"Spec={test_metrics['specificity']:.3f}"
        )

    return {
        "fold": fold,
        "test_subject": fold_info["test_subject"],
        "horizon_sec": horizon_sec,
        "horizon_windows": horizon_windows,
        "test_metrics": test_metrics,
        "latency": latency,
        "threshold": threshold,
        "best_val_auc": best_val_auc,
    }

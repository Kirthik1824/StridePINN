"""
training/baseline_trainer.py — Training logic for classical baselines.

Implements LOSO training and evaluation for:
  - Freezing Index Baseline (Bächlin et al. 2010 proxy)
  - SVM Baseline (Ouyang et al. autoregressive proxy)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from data.dataset import GaitDataset
from training.trainer_utils import prepare_fold_data
from training.prediction_trainer import create_prediction_labels, get_cached_features
from utils import compute_metrics, find_optimal_threshold, compute_lead_time

def train_fogi_baseline_fold(
    fold_info: dict,
    dataset: GaitDataset,
    args,
    device: torch.device,
    logger=None,
) -> dict:
    """
    Evaluates the Freezing Index (FoGI) as a direct predictive threshold.
    """
    fold = fold_info["fold"]
    horizon_sec = getattr(args, "horizon", cfg.default_prediction_horizon)
    window_stride_sec = cfg.window_stride / cfg.target_fs
    horizon_windows = max(1, int(horizon_sec / window_stride_sec))

    all_phys, _ = get_cached_features(dataset, logger)
    # Feature 0 in physics array is FoGI
    
    train_phys = all_phys[fold_info["train_idx"]].numpy()
    test_phys = all_phys[fold_info["test_idx"]].numpy()
    
    test_l = dataset.labels[fold_info["test_idx"]]
    
    train_fogi = train_phys[:, 0]
    test_fogi = test_phys[:, 0]
    
    # We create prediction labels to optimize the threshold
    train_l = dataset.labels[fold_info["train_idx"]]
    train_pred_l = create_prediction_labels(train_l.numpy(), horizon_windows)
    test_pred_l = create_prediction_labels(test_l.numpy(), horizon_windows)
    
    # Simple sigmoid scaling of fogi just to use the same metrics/threshold functions
    # FoGI is strictly positive
    train_probs = 1 / (1 + np.exp(- (train_fogi - 1.0)))
    test_probs = 1 / (1 + np.exp(- (test_fogi - 1.0)))
    
    threshold = find_optimal_threshold(train_pred_l, train_probs)
    test_metrics = compute_metrics(test_pred_l, test_probs, threshold=threshold)
    
    y_pred = (test_probs >= threshold).astype(int)
    
    latency = compute_lead_time(
        test_l.numpy(), y_pred,
        window_stride_sec=window_stride_sec,
        horizon_windows=horizon_windows,
    )

    if logger:
         logger.info(
             f"Fold {fold} [FoGI]: AUC={test_metrics['auc']:.3f}, "
             f"F1={test_metrics['f1']:.3f}, "
             f"Sens={test_metrics['sensitivity']:.3f}"
         )

    return {
        "fold": fold,
        "test_subject": fold_info["test_subject"],
        "horizon_sec": horizon_sec,
        "horizon_windows": horizon_windows,
        "test_metrics": test_metrics,
        "latency": latency,
        "threshold": threshold,
    }

def train_svm_baseline_fold(
    fold_info: dict,
    dataset: GaitDataset,
    args,
    device: torch.device,
    logger=None,
) -> dict:
    """
    Trains an SVM Baseline.
    """
    fold = fold_info["fold"]
    horizon_sec = getattr(args, "horizon", cfg.default_prediction_horizon)
    window_stride_sec = cfg.window_stride / cfg.target_fs
    horizon_windows = max(1, int(horizon_sec / window_stride_sec))

    all_phys, all_ews = get_cached_features(dataset, logger)
    
    # Combine physics and EWS features for SVM
    X_train = np.concatenate([
        all_phys[fold_info["train_idx"]].numpy(),
        all_ews[fold_info["train_idx"]].numpy()
    ], axis=1)
    
    X_test = np.concatenate([
        all_phys[fold_info["test_idx"]].numpy(),
        all_ews[fold_info["test_idx"]].numpy()
    ], axis=1)
    
    test_l = dataset.labels[fold_info["test_idx"]]
    train_l = dataset.labels[fold_info["train_idx"]]
    
    train_pred_l = create_prediction_labels(train_l.numpy(), horizon_windows)
    test_pred_l = create_prediction_labels(test_l.numpy(), horizon_windows)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Subsample training data to speed up SVM training if it's too large, but typically Daphent is small enough
    # If it's too large, we can subsample
    max_samples = 15000
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = train_pred_l[idx]
    else:
        X_train_sub = X_train
        y_train_sub = train_pred_l
        
    # Standard SVC
    model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=cfg.seed)
    
    # In some folds, no FoG events might exist in training (highly unlikely but possible)
    if len(np.unique(y_train_sub)) > 1:
        model.fit(X_train_sub, y_train_sub)
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    else:
        train_probs = np.zeros(len(X_train))
        test_probs = np.zeros(len(X_test))
        
    threshold = find_optimal_threshold(train_pred_l, train_probs) if len(np.unique(train_pred_l)) > 1 else 0.5
    test_metrics = compute_metrics(test_pred_l, test_probs, threshold=threshold)
    
    y_pred = (test_probs >= threshold).astype(int)
    
    latency = compute_lead_time(
        test_l.numpy(), y_pred,
        window_stride_sec=window_stride_sec,
        horizon_windows=horizon_windows,
    )

    if logger:
         logger.info(
             f"Fold {fold} [SVM]: AUC={test_metrics['auc']:.3f}, "
             f"F1={test_metrics['f1']:.3f}, "
             f"Sens={test_metrics['sensitivity']:.3f}"
         )

    return {
        "fold": fold,
        "test_subject": fold_info["test_subject"],
        "horizon_sec": horizon_sec,
        "horizon_windows": horizon_windows,
        "test_metrics": test_metrics,
        "latency": latency,
        "threshold": threshold,
    }

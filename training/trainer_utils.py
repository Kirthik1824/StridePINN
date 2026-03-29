import torch
import numpy as np
import json
from pathlib import Path
from config import cfg
from data.dataset import (
    GaitDataset,
    get_loso_splits,
    normalise_fold,
    apply_normalisation,
    apply_per_subject_normalization,
    make_dataloader,
)
from utils import (
    setup_logger,
    compute_metrics,
    find_optimal_threshold,
    compute_detection_latency,
)

def prepare_fold_data(dataset, fold_info, device, normal_only=False):
    """
    Common data preparation for a LOSO fold.
    Now uses per-subject normalization to account for inter-patient sensor shift.
    """
    # 1. Apply Per-Subject Normalization to the whole dataset slice for this fold
    # This prevents Subject A's bias from affecting Subject B's limit cycle origin.
    
    # Train
    train_idx = fold_info["train_idx"]
    if normal_only:
        train_mask_normal = dataset.labels[train_idx] == 0
        train_idx = train_idx[train_mask_normal.numpy()]
    
    train_w = apply_per_subject_normalization(dataset.windows[train_idx], dataset.subject_ids[train_idx])
    train_a = apply_per_subject_normalization(dataset.ankle[train_idx], dataset.subject_ids[train_idx])
    train_l = dataset.labels[train_idx]
    
    # Val
    val_idx = fold_info["val_idx"]
    val_w = apply_per_subject_normalization(dataset.windows[val_idx], dataset.subject_ids[val_idx])
    val_a = apply_per_subject_normalization(dataset.ankle[val_idx], dataset.subject_ids[val_idx])
    val_l = dataset.labels[val_idx]
    
    # Test
    test_idx = fold_info["test_idx"]
    test_w = apply_per_subject_normalization(dataset.windows[test_idx], dataset.subject_ids[test_idx])
    test_a = apply_per_subject_normalization(dataset.ankle[test_idx], dataset.subject_ids[test_idx])
    test_l = dataset.labels[test_idx]

    # Dummy mean/std to maintain signature compatibility
    dummy_mean = torch.zeros(cfg.num_channels)
    dummy_std = torch.ones(cfg.num_channels)

    return {
        "train": (train_w, train_a, train_l),
        "val": (val_w, val_a, val_l),
        "test": (test_w, test_a, test_l),
        "mean": dummy_mean,
        "std": dummy_std,
    }

def aggregate_and_save_results(all_results, model_name, logger):
    """
    Aggregate metrics across folds and save to JSON.
    """
    metrics_keys = ["auc", "sensitivity", "specificity", "f1"]
    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE RESULTS ({model_name.upper()}, {len(all_results)} folds)")
    logger.info(f"{'='*60}")

    for key in metrics_keys:
        values = [r["test_metrics"][key] for r in all_results]
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            logger.info(f"  {key:15s}: {np.mean(valid):.3f} ± {np.std(valid):.3f} ({len(valid)} valid folds)")
        else:
            logger.info(f"  {key:15s}: N/A (all NaN)")

    latencies = [r["latency"]["median"] for r in all_results if r.get("latency") and not np.isnan(r["latency"]["median"])]
    if latencies:
        logger.info(f"  {'latency (s)':15s}: {np.mean(latencies):.2f} ± {np.std(latencies):.2f}")

    results_path = cfg.results_dir / f"{model_name}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

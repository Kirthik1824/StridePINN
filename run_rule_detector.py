"""
run_rule_detector.py — CLI entry point for Approach 2: Rule-Based FoG Detection.

Usage:
    python run_rule_detector.py --folds 10
    python run_rule_detector.py --folds 10 --sweep
    python run_rule_detector.py --folds 2                   # quick test
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import cfg
from utils import (
    seed_everything, setup_logger,
    compute_metrics, find_optimal_threshold, compute_detection_latency,
)
from data.dataset import GaitDataset, get_loso_splits
from rule_detector import RuleBasedDetector


def parse_args():
    p = argparse.ArgumentParser(
        description="Approach 2: Interpretable Rule-Based FoG Detection"
    )
    p.add_argument("--folds", type=int, default=cfg.num_subjects)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument(
        "--sweep", action="store_true",
        help="Run threshold sweep analysis",
    )
    p.add_argument(
        "--fogi-threshold", type=float, default=cfg.rule_fogi_threshold,
        help="FoGI threshold (normalised)",
    )
    p.add_argument(
        "--radius-threshold", type=float, default=cfg.rule_radius_threshold,
        help="Radius threshold (normalised)",
    )
    p.add_argument(
        "--phase-var-threshold", type=float, default=cfg.rule_phase_var_threshold,
        help="Phase variance threshold (normalised)",
    )
    return p.parse_args()


def run_fold(
    fold_info: dict,
    dataset: GaitDataset,
    args,
    logger=None,
) -> dict:
    """Run one LOSO fold of the rule-based detector."""
    fold = fold_info["fold"]
    train_idx = fold_info["train_idx"]
    test_idx = fold_info["test_idx"]

    train_windows = dataset.windows[train_idx].numpy()
    test_windows = dataset.windows[test_idx].numpy()
    train_labels = dataset.labels[train_idx].numpy()
    test_labels = dataset.labels[test_idx].numpy()

    if logger:
        logger.info(f"Fold {fold}: calibrating on {len(train_windows)} training windows...")

    # Create and calibrate detector
    detector = RuleBasedDetector(
        fogi_threshold=args.fogi_threshold,
        radius_threshold=args.radius_threshold,
        phase_var_threshold=args.phase_var_threshold,
    )
    detector.calibrate(train_windows, fs=cfg.target_fs)

    if logger:
        logger.info(
            f"  Baselines: FoGI={detector._fogi_baseline:.3f}, "
            f"Radius={detector._radius_baseline:.3f}, "
            f"PhaseVar={detector._phase_var_baseline:.3f}"
        )

    # Detect on test set
    predictions, scores = detector.detect_batch(test_windows, fs=cfg.target_fs)

    # Compute metrics using continuous scores
    threshold = find_optimal_threshold(test_labels, scores)
    metrics = compute_metrics(test_labels, scores, threshold=threshold)

    # Detection latency
    y_pred = (scores >= threshold).astype(int)
    window_stride_sec = cfg.window_stride / cfg.target_fs
    latency = compute_detection_latency(test_labels, y_pred, window_stride_sec)

    # Threshold sweep
    sweep_results = None
    if getattr(args, "sweep", False):
        sweep_results = RuleBasedDetector.threshold_sweep(test_labels, scores)

    if logger:
        logger.info(
            f"Fold {fold}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, "
            f"Sens={metrics['sensitivity']:.3f}, Spec={metrics['specificity']:.3f}, "
            f"Recall@thresh={metrics['sensitivity']:.3f}"
        )

    return {
        "fold": fold,
        "test_subject": fold_info["test_subject"],
        "test_metrics": metrics,
        "latency": latency,
        "threshold": threshold,
        "baselines": {
            "fogi": float(detector._fogi_baseline) if detector._fogi_baseline else None,
            "radius": float(detector._radius_baseline) if detector._radius_baseline else None,
            "phase_var": float(detector._phase_var_baseline) if detector._phase_var_baseline else None,
        },
        "sweep": sweep_results,
        "n_train": len(train_labels),
        "n_test": len(test_labels),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "rule_detector",
        log_file=log_dir / "run_rule_detector.log",
    )

    logger.info("Approach 2: Interpretable Rule-Based FoG Detection")
    logger.info(f"  FoGI threshold:      {args.fogi_threshold}")
    logger.info(f"  Radius threshold:    {args.radius_threshold}")
    logger.info(f"  Phase var threshold: {args.phase_var_threshold}")

    dataset = GaitDataset()

    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds:
            break
        res = run_fold(fold_info, dataset, args, logger)
        all_results.append(res)

    # Print summary
    metrics_keys = ["auc", "f1", "sensitivity", "specificity"]
    print(f"\n{'='*60}")
    print(f"  Rule-Based Detector — {len(all_results)}-fold LOSO")
    print(f"{'='*60}")
    for key in metrics_keys:
        vals = [r["test_metrics"][key] for r in all_results]
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            print(f"  {key:15s}: {np.mean(valid):.3f} ± {np.std(valid):.3f}")
        else:
            print(f"  {key:15s}: N/A")

    lat_vals = [
        r["latency"]["median"] for r in all_results
        if r["latency"]["median"] is not None and not np.isnan(r["latency"]["median"])
    ]
    if lat_vals:
        print(f"  {'latency (s)':15s}: {np.mean(lat_vals):.2f} ± {np.std(lat_vals):.2f}")
    print(f"{'='*60}\n")

    # Save results
    results_path = cfg.results_dir / "rule_detector_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove sweep data from saved results (too large)
    save_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "sweep"}
        save_results.append(r_copy)

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Save sweep results separately if requested
    if args.sweep:
        sweep_path = cfg.results_dir / "rule_detector_sweep.json"
        sweep_data = {
            f"fold_{r['fold']}": r.get("sweep", []) for r in all_results
        }
        with open(sweep_path, "w") as f:
            json.dump(sweep_data, f, indent=2)
        logger.info(f"Sweep results saved to {sweep_path}")


if __name__ == "__main__":
    main()

"""
run_prediction.py — CLI entry point for Approach 1: Early Warning FoG Prediction.

Usage:
    python run_prediction.py --horizon 2.0 --epochs 50 --folds 10
    python run_prediction.py --horizon 1.0 --epochs 5 --folds 2    # quick test
    python run_prediction.py --all-horizons --epochs 50 --folds 10  # sweep horizons
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import cfg
from utils import seed_everything, get_device, setup_logger
from data.dataset import GaitDataset, get_loso_splits
from training.prediction_trainer import train_prediction_fold
from training.trainer_utils import aggregate_and_save_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Approach 1: Early Warning FoG Prediction (CNN-LSTM + Physics + EWS)"
    )
    p.add_argument(
        "--horizon", type=float, default=cfg.default_prediction_horizon,
        help=f"Prediction horizon in seconds (default: {cfg.default_prediction_horizon})",
    )
    p.add_argument(
        "--all-horizons", action="store_true",
        help="Run experiments for all configured horizons (1s, 2s, 3s)",
    )
    p.add_argument("--folds", type=int, default=cfg.num_subjects)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--seed", type=int, default=cfg.seed)
    return p.parse_args()


def run_single_horizon(args, dataset, device, logger, horizon):
    """Run LOSO evaluation for a single prediction horizon."""
    args.horizon = horizon

    logger.info(f"\n{'='*60}")
    logger.info(f"Prediction Horizon: {horizon}s")
    logger.info(f"{'='*60}")

    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds:
            break
        res = train_prediction_fold(fold_info, dataset, args, device, logger)
        all_results.append(res)

    # Print summary
    metrics_keys = ["auc", "f1", "sensitivity", "specificity"]
    print(f"\n{'='*60}")
    print(f"  Early Warning Prediction — Horizon {horizon}s — {len(all_results)}-fold LOSO")
    print(f"{'='*60}")
    for key in metrics_keys:
        vals = [r["test_metrics"][key] for r in all_results]
        print(f"  {key:15s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print(f"{'='*60}\n")

    # Save results
    name = f"prediction_h{horizon}s"
    aggregate_and_save_results(all_results, name, logger)

    return all_results


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "prediction",
        log_file=log_dir / "run_prediction.log",
    )

    logger.info(f"Approach 1: Early Warning Prediction, device={device}")
    dataset = GaitDataset()

    if args.all_horizons:
        # Sweep all configured horizons
        all_horizon_results = {}
        for h in cfg.prediction_horizons:
            results = run_single_horizon(args, dataset, device, logger, h)
            all_horizon_results[f"{h}s"] = results

        # Save combined results
        results_path = cfg.results_dir / "prediction_all_horizons.json"
        with open(results_path, "w") as f:
            json.dump(all_horizon_results, f, indent=2, default=str)
        logger.info(f"Combined horizon results saved to {results_path}")
    else:
        run_single_horizon(args, dataset, device, logger, args.horizon)


if __name__ == "__main__":
    main()

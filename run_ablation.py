"""
run_ablation.py — Ablation Study for FoG Prediction

Trains and evaluates 4 variants of the model:
1. Base: CNN-LSTM (raw IMU only)
2. Base + Physics
3. Base + EWS
4. Full Model: CNN-LSTM + Physics + EWS
"""

import argparse
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import cfg
from utils import seed_everything, get_device, setup_logger
from data.dataset import GaitDataset, get_loso_splits
from training.prediction_trainer import train_prediction_fold

def parse_args():
    p = argparse.ArgumentParser(description="Ablation Study for FoG Prediction")
    p.add_argument("--horizon", type=float, default=2.0)
    p.add_argument("--folds", type=int, default=cfg.num_subjects)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--seed", type=int, default=cfg.seed)
    return p.parse_args()

def run_variant(args, dataset, device, logger, name, use_physics, use_ews):
    args.use_physics = use_physics
    args.use_ews = use_ews
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Variant: {name}")
    logger.info(f"{'='*60}")
    
    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds:
            break
        res = train_prediction_fold(fold_info, dataset, args, device, logger)
        all_results.append(res)
        
    metrics_keys = ["auc", "f1", "sensitivity", "specificity"]
    summary = {}
    
    print(f"\n{'='*60}")
    print(f"  {name} — {len(all_results)}-fold LOSO")
    print(f"{'='*60}")
    
    for key in metrics_keys:
        vals = [r["test_metrics"][key] for r in all_results]
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            summary[f"{key}_mean"] = float(np.mean(valid))
            summary[f"{key}_std"] = float(np.std(valid))
            print(f"  {key:15s}: {np.mean(valid):.3f} ± {np.std(valid):.3f} (n={len(valid)})")
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            print(f"  {key:15s}: N/A")
            
    lat_vals = [r["latency"]["median"] for r in all_results if r.get("latency") and not np.isnan(r["latency"]["median"])]
    if lat_vals:
        summary["lead_time_mean"] = float(np.mean(lat_vals))
        summary["lead_time_std"] = float(np.std(lat_vals))
        print(f"  {'lead time (s)':15s}: {np.mean(lat_vals):.2f} ± {np.std(lat_vals):.2f} (n={len(lat_vals)})")
    else:
        summary["lead_time_mean"] = None
        summary["lead_time_std"] = None
        
    print(f"{'='*60}\n")
    return summary, all_results

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "ablation",
        log_file=log_dir / "run_ablation.log",
    )

    dataset = GaitDataset()
    
    variants = [
        {"name": "Base (CNN-LSTM)", "use_physics": False, "use_ews": False},
        {"name": "Base + Physics", "use_physics": True, "use_ews": False},
        {"name": "Base + EWS", "use_physics": False, "use_ews": True},
        {"name": "Full Model", "use_physics": True, "use_ews": True},
    ]
    
    results = {}
    
    for v in variants:
        summary, fold_results = run_variant(
            args, dataset, device, logger, v["name"], v["use_physics"], v["use_ews"]
        )
        results[v["name"]] = {"summary": summary, "folds": fold_results}
        
    results_path = cfg.results_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info(f"\nAblation results saved to {results_path}")

if __name__ == "__main__":
    main()

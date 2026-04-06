import argparse
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import cfg
from utils import seed_everything, get_device, setup_logger
from data.dataset_defog import DeFOGDataset, get_loso_splits
from training.prediction_trainer import train_prediction_fold

def parse_args():
    p = argparse.ArgumentParser(description="DeFOG Ablation Study")
    p.add_argument("--horizon", type=float, default=2.0)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20) # Lower for speed on large dataset
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--seed", type=int, default=cfg.seed)
    return p.parse_args()

def run_variant(args, dataset, device, logger, name, use_physics, use_ews, baseline_type=None):
    args.use_physics = use_physics
    args.use_ews = use_ews
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running DeFOG Variant: {name}")
    logger.info(f"{'='*60}")
    
    all_results = []
    # Use DeFOG's version of get_loso_splits
    for i, fold_info in enumerate(get_loso_splits(dataset, num_folds=args.folds)):
        if baseline_type == "fogi":
            from training.baseline_trainer import train_fogi_baseline_fold
            res = train_fogi_baseline_fold(fold_info, dataset, args, device, logger)
        else:
            res = train_prediction_fold(fold_info, dataset, args, device, logger)
        all_results.append(res)
        
    metrics_keys = ["auc", "f1", "sensitivity", "specificity"]
    summary = {}
    
    for key in metrics_keys:
        vals = [r["test_metrics"][key] for r in all_results]
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            summary[f"{key}_mean"] = float(np.mean(valid))
            summary[f"{key}_std"] = float(np.std(valid))
            
    lat_vals = [r["latency"]["median"] for r in all_results if r.get("latency") and not np.isnan(r["latency"]["median"])]
    if lat_vals:
        summary["lead_time_mean"] = float(np.mean(lat_vals))
        summary["lead_time_std"] = float(np.std(lat_vals))
        
    return summary, all_results

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("defog_ablation", log_file=log_dir / "run_defog_ablation.log")

    dataset = DeFOGDataset(num_subjects=args.folds)
    
    variants = [
        {"name": "Baseline: Freezing Index", "use_physics": False, "use_ews": False, "baseline_type": "fogi"},
        {"name": "Base (CNN-LSTM)", "use_physics": False, "use_ews": False, "baseline_type": None},
        {"name": "Base + Physics", "use_physics": True, "use_ews": False, "baseline_type": None},
        {"name": "Base + EWS", "use_physics": False, "use_ews": True, "baseline_type": None},
        {"name": "Full Model", "use_physics": True, "use_ews": True, "baseline_type": None},
    ]
    
    results = {}
    for v in variants:
        summary, fold_results = run_variant(
            args, dataset, device, logger, v["name"], v["use_physics"], v["use_ews"], v["baseline_type"]
        )
        results[v["name"]] = {"summary": summary, "folds": fold_results}
        
    results_path = cfg.results_dir / "defog_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info(f"\nDeFOG Ablation results saved to {results_path}")

if __name__ == "__main__":
    main()

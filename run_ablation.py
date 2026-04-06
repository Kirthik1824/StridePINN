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
    p.add_argument("--m", type=int, default=2, help="Embedding dimension constraint")
    p.add_argument("--variant", type=str, default="all", help="Specific variant name to run")
    return p.parse_args()

def run_variant(args, dataset, device, logger, name, use_physics, use_ews, baseline_type=None):
    args.use_physics = use_physics
    args.use_ews = use_ews
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Variant: {name}")
    logger.info(f"{'='*60}")
    
    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds:
            break
            
        if baseline_type == "fogi":
            from training.baseline_trainer import train_fogi_baseline_fold
            res = train_fogi_baseline_fold(fold_info, dataset, args, device, logger)
        elif baseline_type == "svm":
            from training.baseline_trainer import train_svm_baseline_fold
            res = train_svm_baseline_fold(fold_info, dataset, args, device, logger)
        else:
            res = train_prediction_fold(fold_info, dataset, args, device, logger)
            
        all_results.append(res)
        
    metrics_keys = ["auc", "f1", "sensitivity", "specificity"]
    summary = {}
    
    print(f"\n{'='*60}")
    print(f"  {name} — {len(all_results)}-fold LOSO (Overall Summary)")
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
        
    print(f"\n  Per-Subject Breakdown:")
    print(f"  {'Subject':10s} | {'AUC':6s} | {'Sensitivity':11s} | {'Lead Time (s)':13s}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*11}-+-{'-'*13}-")
    for r in all_results:
        subj = r["test_subject"]
        auc = r["test_metrics"].get("auc", float("nan"))
        sens = r["test_metrics"].get("sensitivity", float("nan"))
        lead = r["latency"]["median"] if r.get("latency") else float("nan")
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        sens_str = f"{sens:.3f}" if not np.isnan(sens) else "N/A"
        lead_str = f"{lead:.2f}" if not np.isnan(lead) else "N/A"
        print(f"  {str(subj):10s} | {auc_str:6s} | {sens_str:11s} | {lead_str:13s}")
        
    print(f"{'='*60}\n")
    return summary, all_results

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    cfg.delay_embedding_m = args.m

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "ablation",
        log_file=log_dir / f"run_ablation_h{args.horizon}_m{args.m}.log",
    )

    dataset = GaitDataset()
    
    variants = [
        {"name": "Baseline: Freezing Index", "use_physics": False, "use_ews": False, "baseline_type": "fogi"},
        {"name": "Baseline: SVM (Physics+EWS)", "use_physics": False, "use_ews": False, "baseline_type": "svm"},
        {"name": "Base (CNN-LSTM)", "use_physics": False, "use_ews": False, "baseline_type": None},
        {"name": "Base + Physics", "use_physics": True, "use_ews": False, "baseline_type": None},
        {"name": "Base + EWS", "use_physics": False, "use_ews": True, "baseline_type": None},
        {"name": "Full Model", "use_physics": True, "use_ews": True, "baseline_type": None},
    ]

    if args.variant != "all":
        variants = [v for v in variants if v["name"] == args.variant]
        if not variants:
            print(f"Variant '{args.variant}' not found!")
            return
    
    results = {}
    
    for v in variants:
        summary, fold_results = run_variant(
            args, dataset, device, logger, v["name"], v["use_physics"], v["use_ews"], v["baseline_type"]
        )
        results[v["name"]] = {"summary": summary, "folds": fold_results}
        
    results_path = cfg.results_dir / f"sweep_results_h{args.horizon}_m{args.m}_{args.variant.replace(' ', '_').replace('+','_')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info(f"\nAblation results saved to {results_path}")

if __name__ == "__main__":
    main()

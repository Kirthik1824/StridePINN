import argparse
import json
from pathlib import Path
import torch
import numpy as np

from config import cfg
from utils import seed_everything, get_device, setup_logger
from data.dataset import GaitDataset, get_loso_splits

# Trainers
from training.pinn_trainer import train_pinn_fold
from training.supervised_trainer import train_supervised_fold
from training.anomaly_trainer import train_anomaly_fold
from training.hybrid_trainer import train_hybrid_fold
from training.trainer_utils import aggregate_and_save_results

def parse_args():
    p = argparse.ArgumentParser(description="Unified StridePINN Trainer")
    p.add_argument("--model", type=str, required=True, 
                   choices=["cnn", "cnn_lstm", "pinn", "conv_ae", "ocsvm", "hybrid"],
                   help="Model type to train")
    p.add_argument("--folds", type=int, default=cfg.num_subjects)
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--seed", type=int, default=cfg.seed)
    
    # PINN specific
    p.add_argument("--latent_dim", type=int, default=cfg.latent_dim)
    p.add_argument("--lambda_cyc", type=float, default=cfg.lambda_cyc)
    p.add_argument("--lambda_phi", type=float, default=cfg.lambda_phi)
    p.add_argument("--lambda_smooth", type=float, default=cfg.lambda_smooth)
    p.add_argument("--lambda_radius", type=float, default=cfg.lambda_radius)
    
    p.add_argument("--ode_mode", type=str, default=cfg.ode_mode, choices=["mlp", "hopf"],
                   help="ODE vector field mode")
    
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    log_dir = cfg.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"train_{args.model}", log_file=log_dir / f"train_{args.model}.log")

    logger.info(f"Starting training: model={args.model}, device={device}")
    dataset = GaitDataset()
    
    all_results = []
    for i, fold_info in enumerate(get_loso_splits(dataset)):
        if i >= args.folds: break
        
        if args.model == "pinn":
            res = train_pinn_fold(fold_info, dataset, args, device, logger)
        elif args.model in ["cnn", "cnn_lstm"]:
            res = train_supervised_fold(fold_info, dataset, args.model, args, device, logger)
        elif args.model in ["conv_ae", "ocsvm"]:
            res = train_anomaly_fold(fold_info, dataset, args.model, args, device, logger)
        elif args.model == "hybrid":
            res = train_hybrid_fold(fold_info, dataset, args, device, logger)
            
        all_results.append(res)

    aggregate_and_save_results(all_results, args.model, logger)

if __name__ == "__main__":
    main()

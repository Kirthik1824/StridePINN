import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.svm import OneClassSVM
from config import cfg
from utils import save_checkpoint, find_optimal_threshold, compute_metrics, compute_detection_latency
from models.conv_ae import FoGConvAE
from training.trainer_utils import prepare_fold_data
from data.dataset import make_dataloader

def train_anomaly_fold(fold_info, dataset, model_type, args, device, logger):
    fold = fold_info["fold"]
    data = prepare_fold_data(dataset, fold_info, device, normal_only=True)
    
    if model_type == "ocsvm":
        return train_ocsvm(fold_info, data, logger)
    
    train_w, train_a, train_l = data["train"]
    val_w, _, val_l = data["val"]
    test_w, _, test_l = data["test"]
    mean, std = data["mean"], data["std"]

    train_loader = make_dataloader(train_w, train_a, train_l, args.batch_size, shuffle=True)
    model = FoGConvAE(in_channels=cfg.num_channels, latent_dim=64).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_loss, best_state = float('inf'), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            w, _, _ = batch
            x_hat = model(w.to(device))
            loss = criterion(x_hat, w.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss, best_state = avg_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval()

    def get_scores(windows):
        scores = []
        with torch.no_grad():
            for i in range(0, len(windows), args.batch_size):
                batch = windows[i:i+args.batch_size].to(device)
                scores.append(model.anomaly_score(batch).cpu().numpy())
        return np.concatenate(scores)

    val_scores, test_scores = get_scores(val_w), get_scores(test_w)
    opt_thresh = find_optimal_threshold(val_l.numpy(), val_scores)
    test_metrics = compute_metrics(test_l.numpy(), test_scores, threshold=opt_thresh)
    latency = compute_detection_latency(test_l.numpy(), (test_scores >= opt_thresh).astype(int), window_stride_sec=0.8)

    save_checkpoint({"model_state": best_state, "fold": fold, "test_metrics": test_metrics, 
                     "threshold": opt_thresh, "norm_mean": mean, "norm_std": std}, 
                    cfg.checkpoint_dir / "conv_ae" / f"fold_{fold:02d}.pt")

    return {"fold": fold, "test_metrics": test_metrics, "threshold": opt_thresh, "latency": latency}

def train_ocsvm(fold_info, data, logger):
    fold = fold_info["fold"]
    train_w, _, _ = data["train"]
    val_w, _, val_l = data["val"]
    test_w, _, test_l = data["test"]

    train_flat = train_w.reshape(len(train_w), -1).numpy()
    val_flat = val_w.reshape(len(val_w), -1).numpy()
    test_flat = test_w.reshape(len(test_w), -1).numpy()

    if len(train_flat) > 5000:
        train_flat = train_flat[np.random.choice(len(train_flat), 5000, replace=False)]

    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    svm.fit(train_flat)
    
    val_scores = -svm.decision_function(val_flat)
    test_scores = -svm.decision_function(test_flat)
    
    opt_thresh = find_optimal_threshold(val_l.numpy(), val_scores)
    test_metrics = compute_metrics(test_l.numpy(), test_scores, threshold=opt_thresh)
    latency = compute_detection_latency(test_l.numpy(), (test_scores >= opt_thresh).astype(int), window_stride_sec=0.8)

    return {"fold": fold, "test_metrics": test_metrics, "threshold": float(opt_thresh), "latency": latency}

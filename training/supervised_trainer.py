import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import cfg
from utils import save_checkpoint, find_optimal_threshold, compute_metrics, compute_detection_latency
from models.cnn import FoGCNN1D
from models.cnn_lstm import FoGCNNLSTM
from training.trainer_utils import prepare_fold_data
from data.dataset import augment_fog_windows, make_dataloader

def train_supervised_fold(fold_info, dataset, model_type, args, device, logger):
    fold = fold_info["fold"]
    data = prepare_fold_data(dataset, fold_info, device)
    
    train_w, train_a, train_l = data["train"]
    val_w, _, val_l = data["val"]
    test_w, _, test_l = data["test"]
    mean, std = data["mean"], data["std"]

    # Augment
    train_w, train_a, train_l = augment_fog_windows(train_w, train_a, train_l, 
                                                   oversample_factor=cfg.fog_oversample_factor, 
                                                   jitter=cfg.jitter_samples)
    
    n_normal = (train_l == 0).sum().item()
    n_fog = (train_l == 1).sum().item()
    pos_weight = torch.tensor([n_normal / max(n_fog, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = make_dataloader(train_w, train_a, train_l, args.batch_size, shuffle=True)
    
    # Model init
    if model_type == "cnn":
        model = FoGCNN1D(in_channels=cfg.num_channels, conv1_out=cfg.cnn_conv1_out, 
                         conv2_out=cfg.cnn_conv2_out, kernel_size=cfg.cnn_kernel_size, dropout=cfg.cnn_dropout)
    else:
        model = FoGCNNLSTM(in_channels=cfg.num_channels, conv1_out=cfg.lstm_conv1_out, 
                           conv2_out=cfg.lstm_conv2_out, kernel_size=cfg.lstm_kernel_size, 
                           lstm_hidden=cfg.lstm_hidden, dropout=cfg.lstm_dropout)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1, best_state, patience = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            w, _, l = batch
            logits = model(w.to(device))
            loss = criterion(logits, l.to(device).float().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        
        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for i in range(0, len(val_w), args.batch_size):
                batch = val_w[i:i+args.batch_size].to(device)
                val_probs.append(model.predict_proba(batch).cpu().numpy())
        val_probs = np.concatenate(val_probs).squeeze()
        v_metrics = compute_metrics(val_l.numpy(), val_probs)

        if v_metrics["f1"] > best_val_f1:
            best_val_f1, best_state = v_metrics["f1"], {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience: break

    if best_state: model.load_state_dict(best_state)
    model.eval()

    # Test
    test_probs = []
    with torch.no_grad():
        for i in range(0, len(test_w), args.batch_size):
            batch = test_w[i:i+args.batch_size].to(device)
            test_probs.append(model.predict_proba(batch).cpu().numpy())
    test_probs = np.concatenate(test_probs).squeeze()
    
    opt_thresh = find_optimal_threshold(val_l.numpy(), val_probs)
    test_metrics = compute_metrics(test_l.numpy(), test_probs, threshold=opt_thresh)
    latency = compute_detection_latency(test_l.numpy(), (test_probs >= opt_thresh).astype(int), window_stride_sec=0.8)

    save_checkpoint({"model_state": best_state, "fold": fold, "test_metrics": test_metrics, 
                     "threshold": opt_thresh, "norm_mean": mean, "norm_std": std}, 
                    cfg.checkpoint_dir / model_type / f"fold_{fold:02d}.pt")

    return {"fold": fold, "test_metrics": test_metrics, "threshold": opt_thresh, "latency": latency}

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from config import cfg
from utils import save_checkpoint, compute_metrics, compute_detection_latency
from models.pinn import GaitPINN
from training.trainer_utils import prepare_fold_data

def train_hybrid_fold(fold_info, dataset, args, device, logger):
    """
    Train a Hybrid PINN-Classifier:
    1. Load/Train PINN on normal gait.
    2. Extract features (residual, phase, latent stats) from train/val/test.
    3. Train Logistic Regression on these features.
    """
    fold = fold_info["fold"]
    # 1. Prepare data (PINN needs normal only for training, but hybrid needs all for feature extraction)
    # We'll use a pre-trained PINN checkpoint if it exists, otherwise train it.
    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    
    # Re-use prepare_fold_data but get all labels for classifier training
    data = prepare_fold_data(dataset, fold_info, device, normal_only=False)
    train_w, _, train_l = data["train"]
    val_w, _, val_l = data["val"]
    test_w, _, test_l = data["test"]
    
    # 2. Load/Initialize PINN
    model = GaitPINN(
        in_dim=cfg.num_channels,
        latent_dim=args.latent_dim,
        ode_mode=args.ode_mode,
    ).to(device)
    
    if ckpt_path.exists():
        logger.info(f"  Loading pre-trained PINN from {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            logger.info("  Checkpoint loaded successfully.")
        except Exception as e:
            logger.warning(f"  Incompatible checkpoint at {ckpt_path}: {e}")
            logger.info("  Triggering PINN retraining...")
            from training.pinn_trainer import train_pinn_fold
            train_pinn_fold(fold_info, dataset, args, device, logger)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
    else:
        logger.info(f"  No PINN checkpoint found for fold {fold}. Retraining...")
        from training.pinn_trainer import train_pinn_fold
        train_pinn_fold(fold_info, dataset, args, device, logger)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    model.eval()

    # 3. Extract Features
    def extract_features(windows, labels):
        feats = []
        batch_size = args.batch_size
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(device)
                scores = model.compute_anomaly_scores(batch)
                
                # Feature 1: Max Dynamics Residual
                r_max = scores["residual_max"].cpu().numpy()
                # Feature 2: Phase Advance
                p_adv = scores["phase_advance"].cpu().numpy()
                # Feature 3: Latent Variance (measure of 'jitter' in orbit)
                z_traj = scores["z_traj"].permute(1, 0, 2) # (B, T, D)
                z_var = torch.var(z_traj, dim=1).sum(dim=-1).cpu().numpy()
                
                batch_feats = np.stack([r_max, p_adv, z_var], axis=1)
                feats.append(batch_feats)
        
        return np.concatenate(feats), labels.numpy()

    X_train, y_train = extract_features(train_w, train_l)
    X_val, y_val = extract_features(val_w, val_l)
    X_test, y_test = extract_features(test_w, test_l)

    # 4. Train Classifier (Logistic Regression for interpretability)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    test_metrics = compute_metrics(y_test, y_prob)
    latency = compute_detection_latency(y_test, y_pred, window_stride_sec=0.8)

    logger.info(f"  test auc: {test_metrics['auc']:.3f}, f1: {test_metrics['f1']:.3f}, latency: {latency['median']:.2f}s")
    logger.info(f"  coefficients (res, phase, var): {clf.coef_[0]}")

    return {"fold": fold, "test_metrics": test_metrics, "latency": latency, "clf_coef": clf.coef_[0]}

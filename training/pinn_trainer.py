import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import cfg
from utils import save_checkpoint, find_optimal_threshold, compute_metrics, compute_detection_latency
from models.pinn import GaitPINN
from training.trainer_utils import prepare_fold_data

def train_pinn_fold(fold_info, dataset, args, device, logger):
    """
    Train PINN on a single LOSO fold.
    
    Uses a REVERSED curriculum:
      Phase 1 (epochs 1..warmup):     Physics-only (phi + radius + smooth + ode)
      Phase 2 (epochs warmup..ramp):   Ramp up data loss gradually
      Phase 3 (epochs ramp+..end):     Full loss (all terms active)
    """
    fold = fold_info["fold"]
    
    # Determine reconstruction target based on decoder output dimension
    use_all_channels = (cfg.decoder_out == 9)
    data = prepare_fold_data(dataset, fold_info, device, normal_only=True)
    
    train_w, train_a, _ = data["train"]
    val_w, val_a, val_l = data["val"]
    test_w, test_a, test_l = data["test"]
    mean, std = data["mean"], data["std"]

    logger.info(f"  Train (normal only): {len(train_w)} windows")

    # DataLoaders
    from data.dataset import make_dataloader
    train_labels_dummy = torch.zeros(len(train_w), dtype=torch.long)
    train_loader = make_dataloader(train_w, train_a, train_labels_dummy, args.batch_size, shuffle=True)

    # Model
    model = GaitPINN(
        in_dim=cfg.num_channels,
        latent_dim=args.latent_dim,
        encoder_hidden=cfg.encoder_hidden,
        ode_hidden=cfg.ode_hidden,
        decoder_out=cfg.decoder_out,
        ode_method=cfg.ode_method,
        ode_rtol=cfg.ode_rtol,
        ode_atol=cfg.ode_atol,
        ode_step_size=cfg.ode_step_size,
        ode_mode=args.ode_mode,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_loss = float("inf")
    best_state = None

    warmup = cfg.pinn_warmup_epochs    # Physics-only phase
    ramp_steps = cfg.pinn_scheduler_steps  # Data ramp-up phase

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "data": 0, "phi": 0, "smooth": 0, "radius": 0, "ode": 0}
        n_batches = 0

        # ---- Reversed Curriculum Schedule ----
        # Phase 1: Physics only (no data loss) — learn oscillatory latent space
        # Phase 2: Ramp up data loss — shape reconstruction
        # Phase 3: Full loss — fine-tune jointly
        if epoch <= warmup:
            # Physics-only: data weight = 0
            w_data = 0.0
            w_phi = args.lambda_phi
            w_smooth = args.lambda_smooth
            w_radius = getattr(args, 'lambda_radius', cfg.lambda_radius)
            w_ode = 1.0
        elif epoch <= warmup + ramp_steps:
            # Ramp up data loss
            ramp = (epoch - warmup) / ramp_steps
            w_data = ramp
            w_phi = args.lambda_phi
            w_smooth = args.lambda_smooth
            w_radius = getattr(args, 'lambda_radius', cfg.lambda_radius)
            w_ode = 1.0
        else:
            # Full loss
            w_data = 1.0
            w_phi = args.lambda_phi
            w_smooth = args.lambda_smooth
            w_radius = getattr(args, 'lambda_radius', cfg.lambda_radius)
            w_ode = 1.0

        for batch in train_loader:
            windows, ankle, _ = batch
            windows, ankle = windows.to(device), ankle.to(device)
            
            # --- Guide Item 9: Add noise during training for robustness ---
            if model.training:
                windows = windows + 0.01 * torch.randn_like(windows)
            
            # Determine reconstruction target
            if use_all_channels:
                recon_target = windows  # All 9 channels
            else:
                recon_target = ankle    # Just ankle (3 channels)

            try:
                losses = model.compute_loss(
                    windows, recon_target,
                    lambda_phi=w_phi,
                    lambda_smooth=w_smooth,
                    lambda_radius=w_radius,
                )
                
                # Apply data weight (reversed curriculum)
                if w_data < 1.0:
                    # Recompute total with weighted data loss
                    total = (w_data * losses["data"]
                             + w_phi * losses["phi"]
                             + w_smooth * losses["smooth"]
                             + w_radius * losses["radius"]
                             + w_ode * losses["ode"])
                    losses["total"] = total
                
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                for k in epoch_losses:
                    epoch_losses[k] += losses[k].item()
                n_batches += 1
            except RuntimeError as e:
                if "underflow" in str(e).lower() or "overflow" in str(e).lower():
                    logger.warning(f"  Epoch {epoch}: ODE solver error: {e}")
                    continue
                raise

        scheduler.step()
        if n_batches == 0: continue
        avg_loss = epoch_losses["total"] / n_batches

        if epoch % 10 == 0 or epoch == 1:
            phase_str = "PHYSICS" if epoch <= warmup else ("RAMP" if epoch <= warmup + ramp_steps else "FULL")
            
            # Log mean predicted omega across the last batch to monitor adaptation
            with torch.no_grad():
                out_sample = model(windows[:8])
                m_omega = out_sample["omega"].mean().item() / (2.0 * np.pi) # Convert to Hz
            
            logger.info(
                f"  Epoch {epoch:3d} [{phase_str:7s}] — total={avg_loss:.4f}, "
                f"data={epoch_losses['data']/n_batches:.4f}, "
                f"ode={epoch_losses['ode']/n_batches:.4f}, "
                f"hz={m_omega:.2f}, "
                f"radius={epoch_losses['radius']/n_batches:.4f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Threshold Calibration (with fallback for zero-FoG folds)
    val_scores = compute_pinn_scores(model, val_w, device, args.batch_size)
    
    n_fog_val = int(val_l.sum().item())
    feature_keys = ["residual_max", "phase_advance", "var_r", "mean_abs_r_1", "mean_r2"]
    
    def get_X(scores_dict):
        return np.stack([scores_dict[k] for k in feature_keys], axis=1)
        
    X_val = get_X(val_scores)
    y_val = val_l.numpy()
    
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    if n_fog_val > 0:
        clf.fit(X_val_scaled, y_val)
        y_val_prob_raw = clf.predict_proba(X_val_scaled)[:, 1]
        y_val_prob = np.convolve(y_val_prob_raw, np.ones(3)/3, mode='same')
        tau = find_optimal_threshold(y_val, y_val_prob)
    else:
        logger.info(f"  Fold {fold}: 0 FoG in val set, using fallback feature weights")
        clf.fit(np.vstack([np.zeros(5), np.ones(5)]), np.array([0, 1]))
        clf.coef_ = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        clf.intercept_ = np.array([-np.percentile(X_val_scaled.sum(axis=1), 95)])
        y_val_prob_raw = clf.predict_proba(X_val_scaled)[:, 1]
        y_val_prob = np.convolve(y_val_prob_raw, np.ones(3)/3, mode='same')
        tau = float(np.percentile(y_val_prob, 95))

    # Test Eval
    test_scores = compute_pinn_scores(model, test_w, device, args.batch_size)
    X_test_scaled = scaler.transform(get_X(test_scores))
    
    # Predict using Logistic Regression
    y_pred_prob_raw = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Guide Fix: Temporal smoothing to crush transient False Positives
    # Apply a 3-window moving average (reflects ~2.4 seconds of contextual continuity)
    y_pred_prob = np.convolve(y_pred_prob_raw, np.ones(3)/3, mode='same')
    
    y_pred = (y_pred_prob >= tau).astype(int)
    
    test_metrics = compute_metrics(test_l.numpy(), y_pred_prob, threshold=tau)
    latency = compute_detection_latency(test_l.numpy(), y_pred, window_stride_sec=0.8)

    logger.info(f"  TEST — AUC={test_metrics['auc']:.3f}, F1={test_metrics['f1']:.3f}, "
                f"Se={test_metrics['sensitivity']:.3f}, Sp={test_metrics['specificity']:.3f}, "
                f"latency={latency['median']:.2f}s")

    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    save_checkpoint({
        "model_state": best_state, "fold": fold, "test_metrics": test_metrics,
        "tau": tau, "norm_mean": mean, "norm_std": std,
    }, ckpt_path)

    return {"fold": fold, "test_metrics": test_metrics, "tau": tau, "latency": latency}

def compute_pinn_scores(model, windows, device, batch_size):
    model.eval()
    scores = {"residual_max": [], "phase_advance": [], "var_r": [], "mean_abs_r_1": [], "mean_r2": []}
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size].to(device)
            s = model.compute_anomaly_scores(batch)
            for k in scores.keys():
                scores[k].append(s[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in scores.items()}

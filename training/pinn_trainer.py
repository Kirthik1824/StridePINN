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
    """
    fold = fold_info["fold"]
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
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "data": 0, "cyc": 0, "phi": 0, "smooth": 0}
        n_batches = 0

        # Loss Warmup Curriculum
        warmup = cfg.pinn_warmup_epochs
        steps = cfg.pinn_scheduler_steps
        if epoch <= warmup:
            w_cyc = w_phi = w_smooth = 0.0
        else:
            ramp = min(1.0, (epoch - warmup) / steps)
            w_cyc = args.lambda_cyc * ramp
            w_phi = args.lambda_phi * ramp
            w_smooth = args.lambda_smooth * ramp

        for batch in train_loader:
            windows, ankle, _ = batch
            windows, ankle = windows.to(device), ankle.to(device)

            try:
                losses = model.compute_loss(
                    windows, ankle,
                    lambda_cyc=w_cyc,
                    lambda_phi=w_phi,
                    lambda_smooth=w_smooth,
                )
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
            logger.info(f"  Epoch {epoch:3d} — total={avg_loss:.4f}, data={epoch_losses['data']/n_batches:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval()

    # Threshold Calibration
    val_scores = compute_pinn_scores(model, val_w, device, args.batch_size)
    tau_r = find_optimal_threshold(val_l.numpy(), val_scores["residual_max"])
    tau_phi = -find_optimal_threshold(val_l.numpy(), -val_scores["phase_advance"])

    # Test Eval
    test_scores = compute_pinn_scores(model, test_w, device, args.batch_size)
    y_pred = np.maximum((test_scores["residual_max"] >= tau_r).astype(int),
                        (test_scores["phase_advance"] <= tau_phi).astype(int))
    
    test_metrics = compute_metrics(test_l.numpy(), test_scores["residual_max"], threshold=tau_r)
    latency = compute_detection_latency(test_l.numpy(), y_pred, window_stride_sec=0.8)

    logger.info(f"  TEST — AUC={test_metrics['auc']:.3f}, F1={test_metrics['f1']:.3f}, latency={latency['median']:.2f}s")

    ckpt_path = cfg.checkpoint_dir / "pinn" / f"fold_{fold:02d}.pt"
    save_checkpoint({
        "model_state": best_state, "fold": fold, "test_metrics": test_metrics,
        "tau_r": tau_r, "tau_phi": tau_phi, "norm_mean": mean, "norm_std": std,
    }, ckpt_path)

    return {"fold": fold, "test_metrics": test_metrics, "tau_r": tau_r, "tau_phi": tau_phi, "latency": latency}

def compute_pinn_scores(model, windows, device, batch_size):
    model.eval()
    res, phi = [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size].to(device)
            s = model.compute_anomaly_scores(batch)
            res.append(s["residual_max"].cpu().numpy())
            phi.append(s["phase_advance"].cpu().numpy())
    return {"residual_max": np.concatenate(res), "phase_advance": np.concatenate(phi)}

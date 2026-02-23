# System Description Log

A chronological record of the system's state at key milestones.
Each entry is a full snapshot — architecture, configuration, training setup,
and results — followed by a delta from the previous entry.

---

## Entry 1 — 2026-02-23 (Week 1: Initial Working System)

**Status**: First working end-to-end implementation. All three models train
and evaluate under LOSO cross-validation on the Daphnet dataset.

### Dataset

- **Source**: Daphnet Freezing-of-Gait dataset (10 PD subjects)
- **Sensors**: 3 tri-axial accelerometers (ankle, thigh, trunk) → 9 channels
- **Preprocessing**: 7-step pipeline
  1. Label filtering (remove label-1 samples)
  2. Resampling 64 → 40 Hz (polyphase, 8th-order anti-alias LPF at 20 Hz)
  3. Band-pass 0.3–15 Hz (4th-order zero-phase Butterworth via `filtfilt`)
  4. Axis alignment (anteroposterior / vertical / mediolateral)
  5. Sliding window: 128 samples (3.2 s), stride 32 (0.8 s, 75% overlap)
  6. Window labelling: >50% FoG → 1, <10% FoG → 0, else discarded
  7. Channel-wise z-score normalisation (per LOSO fold, train-only stats)
- **Result**: 17,219 windows total (15,058 normal, 2,161 FoG, ~12.5% FoG)

### Model Architectures

#### 1D-CNN (~146k params)
```
Conv1D(9→32, k=8) + BN + ReLU + MaxPool(2)
Conv1D(32→32, k=8) + BN + ReLU + MaxPool(2)
Flatten(1024) → FC(128) + ReLU + Drop(0.3) → FC(32) + ReLU + Drop(0.3) → FC(1) + Sigmoid
```

#### CNN-LSTM (~79k params)
```
Conv1D(9→128, k=4) + ReLU + MaxPool(2)
Conv1D(128→64, k=4) + ReLU + MaxPool(2)
LSTM(64 hidden, 1 layer) → last hidden state
FC(80) + ReLU + Drop(0.3) → FC(40) + ReLU + Drop(0.3) → FC(1) + Sigmoid
```

#### Conv-AE Anomaly (~45k params)
```
Encoder: Conv1D(9→32, k=7, p=3) + BN + ReLU + MaxPool(2)
         Conv1D(32→32, k=7, p=3) + BN + ReLU + MaxPool(2)
         Flatten → FC(64)
Decoder: FC(1152) → Reshape(36, 32) → Interpolate(128, mode=linear)
         Conv1D(32→9, k=7, p=3) + Sigmoid (rescaled to match input range)
```

#### OC-SVM Anomaly (Classical)
```
Flattened Input: 1152-D vector (128 samples × 9 channels)
Kernel: RBF (nu=0.1, gamma='scale')
Training: Subsampled to 5000 normal windows per fold
```

#### PINN (~5k params)
```
Encoder E_φ:  MLP 9 → 64 → 64 → 2  (ReLU)         — maps x(t₀) to z(0)
Neural ODE:   MLP 2 → 32 → 2        (tanh)          — ż = f_θ(z), autonomous
Decoder D_ψ:  Linear 2 → 3                           — reconstructs ankle accel
```
- **ODE solver**: RK4 (fixed step = 0.1), time normalised to [0, 1]
- **Backprop**: Standard (stores forward trajectory)
- **Training paradigm**: Normal-gait only (anomaly detection)
- **FoG detection**: Dynamics residual r(t) = ‖ż_FD − f_θ(z)‖ and phase stagnation ΔΦ
- **Decision rule**: FoG if max(r) > τ_r OR ΔΦ < τ_φ (thresholds via Youden index on val set)

### Loss Function (PINN)

```
L = L_data + λ_cyc · L_cyc + λ_φ · L_φ + λ_s · L_s
```

| Term | Formula | Weight | Purpose |
|:---|:---|:---|:---|
| L_data | MSE(ankle, decoded) | 1.0 | Reconstruction fidelity |
| L_cyc | ‖z(T) − z(0)‖² | λ_cyc = 1.0 | Closed orbit (periodicity) |
| L_φ | Σ max(0, −Δφ_k)² | λ_φ = 10.0 | Phase monotonicity |
| L_s | Σ ‖z̈_k‖² | λ_s = 0.1 | Trajectory smoothness |

### Training Configuration

| Parameter | PINN | Baselines |
|:---|:---|:---|
| Batch size | 1024 | 1024 |
| Epochs | 50 | 50 |
| Optimizer | Adam (lr=1e-3, wd=1e-5) | Adam (lr=1e-3, wd=1e-5) |
| Scheduler | Cosine annealing (T_max=50) | Cosine annealing (T_max=50) |
| Grad clip | 1.0 | 1.0 |
| Loss | 4-term physics loss | Weighted BCE (pos_weight = N_normal/N_FoG) |
| Augmentation | None (normal-only training) | 3× FoG oversample + ±4 sample jitter |
| Early stopping | Best train loss | Val F1, patience 10 |

### Evaluation Protocol

- **LOSO**: 10 folds — 8 train, 1 val (threshold/early-stop), 1 test
- **Metrics**: AUC, Sensitivity, Specificity, F1, Detection Latency (median)
- **Threshold**: Youden index (max Se + Sp − 1) on validation fold

### Full Results (10-fold LOSO)

#### Anomaly Detection Models (train on normal gait only)
| Model | Paradigm | AUC (8 folds) | Se | Sp | F1 |
|:---|:---|:---|:---|:---|:---|
| OC-SVM | Classical | 0.834 ± 0.082 | 0.531 ± 0.446 | 0.802 ± 0.139 | 0.269 ± 0.247 |
| Conv AE | Deep | 0.845 ± 0.078 | 0.528 ± 0.443 | 0.822 ± 0.119 | 0.280 ± 0.257 |
| **PINN** | **Physics** | 0.767 ± 0.081 | 0.449 ± 0.375 | 0.774 ± 0.147 | 0.222 ± 0.213 |

#### Supervised Models (train on normal + FoG)
| Model | AUC (8 folds) | Se | Sp | F1 |
|:---|:---|:---|:---|:---|
| 1D-CNN | 0.921 ± 0.113 | 0.532 ± 0.444 | 0.773 ± 0.333 | 0.349 ± 0.314 |
| CNN-LSTM | 0.924 ± 0.113 | 0.593 ± 0.485 | 0.495 ± 0.449 | 0.216 ± 0.252 |

*AUC computed over 8 valid folds (folds 4 and 10 excluded — no FoG in test set).*

### Observations
1. **Supervised > Anomaly in AUC**: As expected, supervised models (0.92) outperform anomaly detectors (0.77–0.85) when labeled FoG data is available.
2. **Conv AE ≈ OC-SVM > PINN**: Among anomaly detectors, the Conv AE (0.845) and OC-SVM (0.834) currently outperform the PINN (0.767). The PINN's latent dynamics haven't fully converged to limit cycles yet.
3. **High variance across folds**: All models show large std in Se/Sp, reflecting the diverse FoG phenotypes across Daphnet subjects.
4. **PINN needs tuning**: The physics losses (cycle, phase) aren't properly shaping the latent space yet — trajectories collapse to near-fixed-points instead of forming oscillatory limit cycles.



### Known Discrepancies with FYP Report

| Item | Report States | Code Implements | Impact |
|:---|:---|:---|:---|
| ODE solver | DOPRI5 (adaptive) | RK4 (fixed-step) | Speed optimisation; revert for final run |
| Backprop | Adjoint method | Standard odeint | Same reason |
| Batch size | 128 | 1024 | Same reason |
| CNN dropout | Layer 4: Drop(0.5) | Drop(0.3) everywhere | Report typo |
| Integration range | [0, 3.2] s | [0, 1] normalised | Functionally equivalent |
| Δt in residual | 0.025 s | 1/(T−1) ≈ 0.00787 | Scaled; thresholds compensate |

---

<!-- TEMPLATE FOR FUTURE ENTRIES — copy this block and fill in -->
<!--
## Entry N — YYYY-MM-DD (Week X: Title)

**Status**: Brief overall status.

### Changes from Entry N−1

| What Changed | Previous | New | Why |
|:---|:---|:---|:---|
| ... | ... | ... | ... |

### Current Architecture
(Only include sections that changed. Say "Unchanged" for the rest.)

### Results
| Metric | Previous | New |
|:---|:---|:---|
| AUC | ... | ... |
| Sensitivity | ... | ... |
| Specificity | ... | ... |
| F1 | ... | ... |
| Latency | ... | ... |

### Observations / Next Steps
- ...
-->

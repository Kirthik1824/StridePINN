# StridePINN: Physics-Augmented Freezing of Gait Detection

This repository implements three physics-informed approaches for detecting Freezing of Gait (FoG) using wearable sensor data (Daphnet dataset). By modeling normal human gait as a stable limit-cycle oscillator, FoG episodes can be detected as chaotic breakdowns of this cycle.

## Three Approaches

We explore three distinct paths to integrating physics into FoG detection. Each approach is developed in its own isolated branch from `main`:

### Approach 1: Signal-Space Physics Features (Branch: `approach-1/physics-features`)
Extracts interpretable, deterministic physics biomarkers from each window, including:
- **Freeze Index (FoGI):** Ratio of PSD power in 3–8 Hz (tremor) vs 0.5–3 Hz (locomotion)
- **Delay Embedding (Limit Cycle):** Radius variance and phase advance over the delay-embedded orbit
- **Cadence & Energy:** Auto-correlation step regularity
- *Classification:* Support Vector Machine (SVM) and Logistic Regression over a 10-fold LOSO scheme.

### Approach 2: Physics-Augmented Deep Sequence Model (Branch: `approach-2/hybrid-deep-model`)
Fuses deep learning with physical priors:
- **CNN-LSTM + Physics Fusion:** An existing CNN-LSTM backbone concatenated with the engineered physics features (FoGI, radius, etc.) before the final classifier.
- **GRU-PINN:** An autoencoder (unsupervised representation learning) that maps 9D IMU data to a 2D latent space. Regularized with physics-informed losses: cycle closure ($z_T \approx z_0$), temporal smoothness, and (optional) Hopf ODE normal form dynamics.

### Approach 3: Dynamical Mode Decomposition (Branch: `approach-3/dmd-analysis`)
A training-free dynamical systems approach:
- **DMD Triple Index:** Evaluates how well a window fits a few sinusoidal modes (via Hankel matrix SVD).
- **Reconstruction Error:** A measure of periodicity loss during FoG.
- *Classification:* Logistic regression on extracted DMD features.

## Data & Preprocessing
- Dataset: Daphnet Freezing of Gait dataset (10 subjects, wearable accelerometers).
- Pipeline: Available in `data/preprocess.py` (downsampling to 40 Hz, band-pass filtering, 128-sample sliding window).

## Environment Setup
```bash
pip install -r requirements.txt
```

*(Note: Approach 3 Optionally uses PyDMD if installed, but falls back to exact SVD implementations).*

## Evaluating the Approaches
To test any approach, check out its branch:
```bash
git checkout <branch_name>
```
Follow the specific scripts contained within each branch (`classify_physics.py` for Approach 1, `train_approach2.py` for Approach 2, `evaluate_dmd.py` for Approach 3).

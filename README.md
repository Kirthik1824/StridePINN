# StridePINN: Physics-Informed Gait Modeling for FoG Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

StridePINN is a research project implementing a **Physics-Informed Neural Network (PINN)** framework for detecting **Freezing of Gait (FoG)** in Parkinson's Disease patients using wearable inertial sensors.

Instead of treating FoG as a standard signal-pattern classification problem, StridePINN models human gait as an orbitally stable **limit cycle** in a 2D latent space. FoG episodes are detected as physical anomalies—departures from the learned dynamical law—manifesting as dynamics residuals and phase stagnation.

## 🚀 Key Features

- **Physics-Informed Latent Dynamics**: Learns gait as a 2D Neural ODE system governed by biomechanical constraints.
- **Neural ODEs**: Continuous-time dynamics modeling using `torchdiffeq` with Adjoint Sensitivity Method for efficient backpropagation.
- **Four-term Physics Loss**:
  - **Reconstruction**: Grounding latent states in ankle acceleration.
  - **Limit-Cycle (Periodicity)**: Enforcing orbit closure ($z(T) \approx z(0)$).
  - **Phase Monotonicity**: Ensuring strictly forward rotation in the latent plane.
  - **Smoothness**: Penalizing high latent acceleration.
- **Interpretable Detection**: FoG is flagged via two physical signals:
  - **Dynamics Residual $r(t)$**: Spikes when the trajectory departs from the learned vector field.
  - **Phase Stagnation $\Delta\Phi$**: Drops when the "clock" of the gait cycle stops.
- **Comprehensive Benchmarks**: Includes supervised 1D-CNN and CNN-LSTM baselines for direct performance comparison.

## 📊 Methodology: Gait as a Limit Cycle

Human walking is fundamentally rhythmic. In healthy gait, the trajectory of body segments traces a periodic orbit (limit cycle). Parkinsonian freezing disrupts this rhythm. 

StridePINN uses a **Self-Supervised Anomaly Detection** paradigm:
1. The model is trained exclusively on **normal-gait windows** to learn the "law of healthy walking."
2. At inference, it monitors how well the current sensor data fits that law.
3. If the physics "breaks" (e.g., the phase stops advancing or the residual spikes), a freeze is detected.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Kirthik1824/StridePINN.git
cd StridePINN

# Install dependencies
pip install -r requirements.txt
```

## 📂 Project Structure

```
StridePINN/
├── config.py              # Global hyperparameters and configuration
├── utils.py               # Shared helpers (logging, metrics, device)
├── data/
│   ├── download_daphnet.py  # Automated download of UCI Daphnet dataset
│   ├── preprocess.py        # 7-step signal processing pipeline
│   └── dataset.py           # PyTorch Dataset and LOSO split logic
├── models/
│   ├── cnn.py               # 1D-CNN Baseline (~146k params)
│   ├── cnn_lstm.py           # CNN-LSTM Baseline (~79k params)
│   └── pinn.py              # Physics-Informed Neural ODE (~5k params)
├── train_baselines.py     # Supervised training for CNN/LSTM
├── train_pinn.py          # Physics-informed training for PINN
├── evaluate.py            # Model comparison and diagnostics
└── visualize.py           # Latent space plots, residuals, and ROC curves
```

## 📖 Usage Guide

### 1. Data Preparation
Download the Daphnet FoG dataset (UCI) and run the 7-step preprocessing pipeline (resampling, filtering, axis alignment, etc.):
```bash
python3 data/download_daphnet.py
python3 data/preprocess.py
```

### 2. Training Baselines
Train the supervised benchmarks using Leave-One-Subject-Out (LOSO) cross-validation:
```bash
python3 train_baselines.py --model cnn
python3 train_baselines.py --model cnn_lstm
```

### 3. Training the PINN
Train the Physics-Informed model on normal gait data:
```bash
python3 train_pinn.py
```

### 4. Evaluation and Visualization
Generate performance tables and visualize the latent gait dynamics:
```bash
python3 evaluate.py
python3 visualize.py --fold 1
```

## 📈 Results (Preliminary)

The system achieves competitive performance on the Daphnet dataset:
- **Baseline AUC**: ~0.92+ across 10-fold LOSO.
- **Detection Latency**: Median 0.0s (immediate detection at episode onset).
- **Efficiency**: The PINN uses ~30x fewer parameters than standard deep learning models while remaining physically interpretable.

---

## 🔗 References

- Chen et al., 2018. *"Neural Ordinary Differential Equations"* (NeurIPS)
- Raissi et al., 2019. *"Physics-informed neural networks"* (J. Comput. Phys.)
- Bachlin et al., 2009. *"Daphnet Freezing of Gait Dataset"* (UCI)

# StridePINN: Multi-Paradigm FoG Detection with Physics-Informed Neural ODEs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

StridePINN is a research framework for detecting **Freezing of Gait (FoG)** in Parkinson's Disease patients using wearable inertial sensors. It implements a **three-tier comparative framework**, evaluating FoG detection across supervised classification, unsupervised anomaly detection, and a novel **Physics-Informed Neural Network (PINN)** paradigm.

## 🚀 Key Features

- **Multi-Paradigm Evaluation**: Comparative analysis across three detection philosophies:
  - **Supervised**: High-capacity deep learning (CNN, CNN-LSTM).
  - **Unsupervised Anomaly**: Reconstruction-based detection (Conv Autoencoder, One-Class SVM).
  - **Physics-Informed**: Dynamics-based detection via latent Neural ODEs.
- **Physics-Informed Latent Dynamics**: Learns gait as a 2D Neural ODE system governed by limit-cycle biomechanical constraints.
- **Neural ODEs**: Continuous-time dynamics modeling using `torchdiffeq` with Adjoint Sensitivity Method.
- **Interpretability**: PINN flags FoG via clinically relevant biomarkers:
  - **Dynamics Residual $r(t)$**: Spikes when the trajectory departs from the learned "law of walking."
  - **Phase Stagnation $\Delta\Phi$**: Plateaus when the gait rhythm collapses.
  - **Continuous Analysis**: New interpretability suite overlays dynamics on ground-truth FoG timelines.
- **Label-Free PINN Training**: The PINN and anomaly baselines require **no FoG annotations** for training—learning exclusively from normal gait data.

## 📊 Methodology: Three Pillars of Detection

| Paradigm | Models | Labels Required | Key Differentiator |
|:---|:---|:---|:---|
| **Supervised** | 1D-CNN, CNN-LSTM | Normal + FoG | Performance Ceiling |
| **Deep Anomaly** | Conv Autoencoder | Normal Only | Reconstruction Error |
| **Physics Anomaly** | **StridePINN** | **Normal Only** | **Dynamical Interpretability** |

StridePINN focuses on the **Physics Anomaly** pillar, modeling gait as an orbitally stable limit cycle. FoG is treated as a dynamical collapse—a loss of oscillator stability that can be detected without ever seeing a labeled "freeze" during training.

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
├── train.py               # Unified entry point for all models
├── evaluate.py            # Aggregate 5-model benchmark
├── utils.py               # Shared utility functions
├── data/
│   ├── preprocess.py      # 7-step signal processing pipeline
│   ├── dataset.py         # PyTorch Dataset and LOSO split logic
│   └── download_daphnet.py # Script to fetch raw data
├── models/
│   ├── cnn.py             # 1D-CNN Supervised (~146k params)
│   ├── cnn_lstm.py        # CNN-LSTM Supervised (~79k params)
│   ├── conv_ae.py         # Conv Autoencoder Anomaly (~45k params)
│   └── pinn.py            # StridePINN Physics-Informed (~5k params)
├── training/
│   ├── supervised_trainer.py # Training logic for CNN/LSTM
│   ├── anomaly_trainer.py    # Training logic for ConvAE/OCSVM
│   ├── pinn_trainer.py       # Training logic for StridePINN
│   └── trainer_utils.py      # Data preparation and logging utils
├── visualization/
│   ├── visualize.py       # Latent phase-plane and residual plots
│   └── interpretability.py # Subject-level interpretability traces
└── scripts/
    └── run_ablations.py   # Automated physics loss ablation suite
```

## 📈 Results (Aggregate LOSO Metrics)

| Model | Paradigm | AUC | Se | Sp | F1 |
|:---|:---|:---|:---|:---|:---|
| CNN-LSTM | Supervised | **0.958** | 0.813 | 0.822 | 0.542 |
| Conv AE | Deep Anomaly | 0.845 | 0.528 | 0.822 | 0.280 |
| **PINN** | **Physics** | 0.552 | 0.449 | 0.774 | 0.222 |

*\* PINN aggregate results show high inter-subject variance; peak performance on Fold 1 reaches **0.813 AUC**.*

---

## 🔗 References

- Chen et al., 2018. *"Neural Ordinary Differential Equations"* (NeurIPS)
- Raissi et al., 2019. *"Physics-informed neural networks"* (J. Comput. Phys.)
- Bachlin et al., 2009. *"Daphnet Freezing of Gait Dataset"* (UCI)
- Sigcha et al., 2024. *"Deep learning for FoG detection: a cross-dataset study"* (ESWA)

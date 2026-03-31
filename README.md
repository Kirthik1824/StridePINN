# StridePINN — Physics-Informed Freezing of Gait Detection & Prediction

FoG (Freezing of Gait) is not just an event — it is a **dynamical collapse of a limit-cycle**, and this collapse can be **predicted before it happens**.

## Research Approaches

### Approach 1: Early Warning Prediction (Main Paper)

> **"Early Warning Signals of Limit-Cycle Collapse in Human Gait"**

Predicts FoG **before onset** using dynamical system early-warning signals:
- CNN-LSTM backbone + physics feature fusion + early-warning signals
- Configurable prediction horizons (1s, 2s, 3s)
- LOSO cross-validation on Daphnet dataset

```bash
python run_prediction.py --horizon 2.0 --epochs 50 --folds 10
python run_prediction.py --all-horizons --epochs 50 --folds 10
```

### Approach 2: Interpretable Rule-Based Detection (Contingency)

> **"A Fully Interpretable, Training-Free System for FoG Detection Using Dynamical Signatures"**

Deterministic, training-free detection using physics rules:
- FoGI threshold, radius collapse, phase instability
- Subject-normalised for cross-patient generalisation
- Prioritises recall (safety-critical: missing a freeze > false alarm)

```bash
python run_rule_detector.py --folds 10
python run_rule_detector.py --folds 10 --sweep
```

## Project Structure

```
StridePINN/
├── config.py                    # Central configuration
├── features.py                  # Physics feature extraction (shared)
├── features_ews.py              # Early Warning Signal features (Approach 1)
├── utils.py                     # Shared utilities (metrics, logging)
├── rule_detector.py             # Approach 2: Rule-based detector
├── run_prediction.py            # Approach 1: CLI entry point
├── run_rule_detector.py         # Approach 2: CLI entry point
├── data/
│   ├── download_daphnet.py      # Dataset download script
│   ├── preprocess.py            # 7-step preprocessing pipeline
│   ├── dataset.py               # PyTorch Dataset + LOSO splits
│   └── signal_physics.py        # Signal-level physics computations
├── models/
│   └── cnn_lstm_prediction.py   # CNN-LSTM + EWS prediction model
├── training/
│   ├── trainer_utils.py         # LOSO fold data preparation
│   └── prediction_trainer.py    # Approach 1 training loop
└── visualization/
    ├── prediction_plots.py      # Phase portraits, FoGI trajectories
    └── rule_plots.py            # Threshold sweeps, ROC/PR curves
```

## Setup

```bash
pip install -r requirements.txt
python data/download_daphnet.py    # Download Daphnet FoG dataset
python data/preprocess.py          # Run preprocessing pipeline
```

## Dataset

Uses the [Daphnet Freezing of Gait Dataset](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait) — 10 subjects, 9-axis accelerometer data (ankle, thigh, trunk), 64 Hz → resampled to 40 Hz.

## Key Insight

Normal gait is a **stable limit-cycle oscillator**. FoG represents a **critical transition** — a collapse of this cycle. Before collapse, dynamical systems exhibit measurable early-warning signals: increased variance, loss of periodicity, phase instability.

This project exploits that insight for both **prediction** (Approach 1) and **interpretable detection** (Approach 2).

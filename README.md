# StridePINN: Dynamical Limit-Cycle Early Warning for Freezing of Gait

**StridePINN** is a research framework demonstrating that Freezing of Gait (FoG) in Parkinson's Disease can be modeled as a **critical transition out of an oscillatory limit-cycle attractor**.

By extracting Takens delay-embedding features and Early Warning Signals (EWS), this framework predicts FoG events **before** onset, moving beyond traditional post-onset classification.

---

## Core Concept
Normal walking is a stable **limit cycle** in reconstructed phase space. FoG is a **critical collapse** of that attractor. Before collapse, the system exhibits measurable Early Warning Signals: increased state-space variance, slowing of phase velocity, and autocorrelation decay.

We fuse these physics-derived features with a CNN-LSTM backbone to predict freezes 1.6 seconds before onset.

## Key Results (10-Fold LOSO)

### Daphnet Dataset (n=10 subjects, 9-channel leg sensors)
| Configuration | AUC | Lead Time (s) | Sensitivity |
| :--- | :--- | :--- | :--- |
| Baseline: Freezing Index | 0.746 ± 0.067 | 0.85 ± 0.71 | 0.581 |
| Baseline: SVM | 0.823 ± 0.098 | 1.20 ± 0.57 | 0.590 |
| CNN-LSTM (Base) | 0.859 ± 0.080 | 1.05 ± 0.53 | 0.637 |
| **Base + Physics** | **0.871 ± 0.079** | **1.30 ± 0.56** | **0.654** |
| Base + EWS | 0.858 ± 0.054 | 1.15 ± 0.55 | 0.632 |
| Full Model | 0.869 ± 0.059 | 1.00 ± 0.57 | 0.619 |

**Finding:** Deterministic physics features (FoGI, delay-embedding) provide the strongest early warning. Adding stochastic EWS features improves specificity (0.896) and F1 (0.459) but *reduces* lead time — a specificity-sensitivity tradeoff. See `docs/manuscript.md` for the full analysis.

### DeFOG Dataset (trunk sensor, cross-dataset feasibility)
The framework's Base+Physics model achieves a 1.40s lead time across a 7-session subsample of the DeFOG dataset (AUC > 0.90), providing preliminary feasibility evidence that limit-cycle collapse features are observable at the body center.

## Repository Structure
```
StridePINN/
├── data/
│   ├── dataset.py           # Daphnet GaitDataset with LOSO splits
│   ├── dataset_defog.py     # DeFOG/tDCS preprocessing and loader
│   ├── preprocess.py        # Butterworth filtering & sliding windows
│   └── download_daphnet.py  # Script to download Daphnet dataset
├── models/
│   ├── cnn_lstm_prediction.py # 3-stream hybrid architecture (Raw + Physics + EWS)
│   └── __init__.py
├── training/
│   ├── prediction_trainer.py # LOSO training loop with shifted labels
│   ├── baseline_trainer.py  # SVM and FoGI baseline trainers
│   └── trainer_utils.py     # Shared utilities
├── visualization/
│   ├── ablation_plots.py    # Ablation bar charts
│   ├── combined_plots.py    # Cross-dataset comparison figure
│   └── prediction_plots.py  # Phase portrait and FoGI trajectory visualizer
├── docs/
│   └── manuscript.md        # Formal academic manuscript
├── results/
│   ├── ablation_results.json      # Daphnet 10-fold LOSO results
│   ├── defog_ablation_results.json # DeFOG feasibility test results
│   └── figures/                    # Publication figures
├── features.py              # Physics feature extraction (FoGI, delay embedding)
├── features_ews.py          # EWS feature extraction (critical transitions)
├── run_ablation.py          # Daphnet ablation study runner
├── run_defog_ablation.py    # DeFOG ablation study runner
├── config.py                # Hyperparameter configuration
├── utils.py                 # Metrics, lead time computation, utilities
└── requirements.txt
```

## Running the Experiments

### 1. Daphnet Ablation Study
```bash
python3 run_ablation.py --epochs 50 --folds 10
```

### 2. DeFOG Cross-Dataset Feasibility Test
```bash
# Step 1: Preprocess (downloads must be in data/raw/defog and data/raw/tdcsfog)
python3 data/dataset_defog.py

# Step 2: Run ablation
python3 run_defog_ablation.py --epochs 20 --folds 10
```

### 3. Visualization
```bash
python3 visualization/ablation_plots.py
python3 visualization/combined_plots.py
python3 visualization/prediction_plots.py
```

## Conclusion
Explicit physics-informed feature engineering extends the early warning lead time for FoG prediction. The framework serves as a proof-of-concept for continuous state-space stability monitoring -- not a production clinical device. Sensitivity (~65%) and sample size ($n=10$) remain the primary limitations.

For full details, see [`docs/manuscript.md`](docs/manuscript.md).

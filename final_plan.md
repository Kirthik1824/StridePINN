# Physics-Grounded Analysis of Freezing of Gait (FoG)

## Final Research Plan (1-Month Execution)

---

# 1. Objective

To investigate whether **human gait can be modeled as a limit-cycle system** and whether **Freezing of Gait (FoG) corresponds to deviations from this cycle**, using:

* Signal-space physics (deterministic, verifiable)
* Physics-informed neural models (evaluated, not assumed)
* Synthetic dynamical systems (for theoretical validation)

---

# 2. Core Hypothesis

### Theoretical Model

Human gait can be approximated as a **limit-cycle oscillator**:

[
\dot{r} = \mu r - r^3, \quad \dot{\phi} = \omega
]

Where:

* ( r ): amplitude (stride vigor)
* ( \phi ): phase (gait progression)
* ( \omega ): cadence
* ( \mu ): stability parameter

---

### Interpretation

| State           | Behavior                                           |
| --------------- | -------------------------------------------------- |
| Normal gait     | Stable limit cycle (constant radius, steady phase) |
| FoG (akinesia)  | Collapse of radius (r → 0)                         |
| FoG (trembling) | Frequency shift (3–8 Hz oscillations)              |

---

# 3. Research Strategy (Three Parallel Tracks)

---

## Track A — PINN Architecture Evaluation (Exploratory)

### Goal

Evaluate whether a physics-informed neural model can learn gait dynamics.

### Key Change

Replace framewise encoder with sequence encoder:

```python
GRU → latent trajectory z(t)
```

### Expected Outcomes

| Outcome      | Interpretation                           |
| ------------ | ---------------------------------------- |
| Learns orbit | Successful latent dynamics modeling      |
| Collapses    | Confirms difficulty of learning dynamics |

---

### Additional Experiment

Perform **loss ablation study**:

| Config     | Components          |
| ---------- | ------------------- |
| Data only  | Reconstruction      |
| + L_phi    | Phase monotonicity  |
| + L_radius | Circular constraint |
| + L_ode    | Vector field        |
| Full       | All losses          |

---

### Contribution

* Quantifies impact of physics constraints
* Identifies failure modes of PINNs in biomedical signals

---

## Track B — Signal-Space Physics (Primary Contribution)

### Principle

> Instead of learning physics → directly measure it from signals

---

### Features to Compute

#### 1. Freeze Index (FoGI)

[
FoGI = \frac{P_{3-8Hz}}{P_{0.5-3Hz}}
]

* Measures breakdown of locomotion rhythm

---

#### 2. Delay Embedding (Takens)

Construct:
[
(x(t), x(t+\tau))
]

* Reconstructs phase space from IMU signal

---

#### 3. Radius (Limit-Cycle Amplitude)

[
r = \sqrt{x^2 + y^2}
]

* Normal: stable
* FoG: collapse

---

#### 4. Phase Advance

[
\phi = \tan^{-1}(y/x)
]

* Normal: steady increase
* FoG: stagnation

---

### Implementation (Deterministic)

```python
from scipy.signal import welch
import numpy as np

def compute_features(signal, fs=40):
    freqs, psd = welch(signal, fs=fs)

    freeze = psd[(freqs>=3)&(freqs<=8)].sum()
    loco   = psd[(freqs>=0.5)&(freqs<=3)].sum()
    fogi = freeze / (loco + 1e-8)

    tau = 5
    x = signal[:-tau]
    y = signal[tau:]

    r = np.sqrt(x**2 + y**2)
    phi = np.unwrap(np.arctan2(y,x))
    dphi = np.diff(phi)

    return [
        fogi,
        np.var(r),
        np.mean(r),
        np.mean(dphi),
        np.std(dphi)
    ]
```

---

### Model

* Logistic Regression / SVM
* LOSO validation (same as baseline)

---

### Expected Results

| Feature | Behavior in FoG |
| ------- | --------------- |
| FoGI    | ↑ increases     |
| Radius  | ↓ collapses     |
| Phase   | ↓ stagnates     |

---

### Key Output

#### Phase Portraits

Plot:

```
x(t) vs x(t+τ)
```

* Normal → circular orbit
* FoG → collapse/noise

---

## Track C — Synthetic Validation (Theoretical Proof)

### Goal

Verify hypothesis under controlled conditions

---

### Method

Simulate stochastic Hopf oscillator:

[
dr = (\mu r - r^3)dt + \sigma dW_t
]

---

### Experiments

1. Clean system (σ = 0)

   * Perfect orbit

2. Add noise

   * Orbit degradation

3. Introduce bifurcation (μ < 0)

   * Collapse (FoG)

---

### Output

* AUC vs noise level
* Visualization of orbit breakdown

---

### Contribution

* Confirms hypothesis independently of real data
* Explains failure of neural models under noise

---

# 4. Experimental Pipeline

---

## Dataset

* Daphnet FoG dataset
* LOSO (Leave-One-Subject-Out)

---

## Models Compared

| Model          | Type              |
| -------------- | ----------------- |
| CNN-LSTM       | Supervised        |
| Conv-AE        | Anomaly detection |
| PINN           | Physics-informed  |
| Signal Physics | Proposed          |

---

## Metrics

* AUC
* Sensitivity / Specificity
* F1-score
* Detection latency

---

# 5. Expected Contributions

---

## Contribution 1 — Theoretical

* Formal mapping of gait → limit cycle system

---

## Contribution 2 — Empirical

* Signal-space validation of limit-cycle behavior
* Direct visualization of FoG as collapse

---

## Contribution 3 — Modeling Insight

* Demonstration that PINNs struggle with IMU dynamics
* Identification of failure modes (mode collapse, desynchronization)

---

## Contribution 4 — Practical

* Physics-based features for FoG detection
* Interpretable biomarkers

---

# 6. Timeline (4 Weeks)

---

## Week 1

* Implement signal features
* Generate phase portraits
* Validate FoGI

---

## Week 2

* Run LOSO experiments
* Perform PINN ablation

---

## Week 3

* Synthetic simulations
* Generate final plots

---

## Week 4

* Write paper
* Compile results

---

# 7. Final Positioning

---

## What this work claims

* Gait exhibits limit-cycle structure (validated)
* FoG corresponds to breakdown of this structure
* Physics can be measured reliably in signal space

---

## What this work does NOT claim

* Perfect latent dynamical modeling
* Fully solved FoG detection problem

---

# 8. Key Insight

> Physics exists clearly in signal space,
> but learning it in latent space from IMU data is fundamentally difficult.

---

# 9. Outcome

This work guarantees:

* A publishable IEEE-level contribution
* A theoretically grounded narrative
* Fully verifiable experimental results (no reliance on uncertain models)

---

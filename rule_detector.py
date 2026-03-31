"""
rule_detector.py — Interpretable Rule-Based FoG Detection (Approach 2).

A fully deterministic, training-free system that detects FoG using
physics-derived dynamical signatures:
  - FoGI (freeze index) elevation
  - Radius collapse (delay-embedding)
  - Phase variance increase

Key properties:
  - Zero training required
  - Fully interpretable decisions
  - Prioritises recall over precision (safety-critical)
  - Subject-normalised for cross-patient generalisation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import cfg
from features import (
    compute_fogi,
    compute_delay_embedding_features,
    compute_signal_energy,
    compute_cadence_regularity,
)


class RuleBasedDetector:
    """
    Deterministic FoG detector based on physics rules.

    Rules:
      1. FoGI > threshold → freeze-band trembling detected
      2. Radius mean < threshold → limit-cycle collapsed
      3. Phase variance > threshold → phase instability

    Scoring:
      score = w1 * fogi_rule + w2 * radius_rule + w3 * phase_rule
      where each rule ∈ {0, 1} and weights sum to 1.

    For LOSO evaluation, thresholds can optionally be calibrated
    on training subjects using percentile-based normalization.
    """

    def __init__(
        self,
        fogi_threshold: float = None,
        radius_threshold: float = None,
        phase_var_threshold: float = None,
        weights: Tuple[float, ...] = None,
        score_threshold: float = 0.3,
    ):
        self.fogi_threshold = fogi_threshold or cfg.rule_fogi_threshold
        self.radius_threshold = radius_threshold or cfg.rule_radius_threshold
        self.phase_var_threshold = phase_var_threshold or cfg.rule_phase_var_threshold
        self.weights = weights or cfg.rule_scoring_weights
        self.score_threshold = score_threshold

        # Normalisation stats (set via calibrate())
        self._fogi_baseline = None
        self._radius_baseline = None
        self._phase_var_baseline = None

    def extract_rule_features(
        self,
        window: np.ndarray,
        fs: int = 40,
        tau: int = 5,
        channel: int = 0,
    ) -> Dict[str, float]:
        """
        Extract the 3 features used by the rule system.

        Args:
            window: (128, C) array
            fs: sampling rate
            tau: delay embedding lag
            channel: which channel

        Returns:
            Dict with fogi, radius_mean, dphi_std
        """
        sig = window[:, channel] if window.ndim > 1 else window

        fogi = compute_fogi(sig, fs)
        embed_feats = compute_delay_embedding_features(sig, tau)

        return {
            "fogi": fogi,
            "radius_mean": embed_feats["r_mean"],
            "dphi_std": embed_feats["dphi_std"],
            "energy": compute_signal_energy(sig),
            "cadence_reg": compute_cadence_regularity(sig, fs),
        }

    def calibrate(
        self,
        windows: np.ndarray,
        fs: int = 40,
        tau: int = 5,
        channel: int = 0,
    ):
        """
        Calibrate normalisation baselines from training data.

        Computes median values of each feature across training windows
        to enable subject-independent normalisation.
        """
        fogis = []
        radii = []
        phase_vars = []

        for i in range(len(windows)):
            feats = self.extract_rule_features(windows[i], fs, tau, channel)
            fogis.append(feats["fogi"])
            radii.append(feats["radius_mean"])
            phase_vars.append(feats["dphi_std"])

        self._fogi_baseline = np.median(fogis)
        self._radius_baseline = np.median(radii)
        self._phase_var_baseline = np.median(phase_vars)

    def score_window(self, features: Dict[str, float]) -> float:
        """
        Compute a detection score for a single window.

        Returns: score ∈ [0, 1] where higher = more likely FoG.
        """
        fogi = features["fogi"]
        radius = features["radius_mean"]
        dphi_std = features["dphi_std"]

        # Normalise relative to baseline if calibrated
        if self._fogi_baseline is not None and self._fogi_baseline > 0:
            fogi = fogi / self._fogi_baseline
        if self._radius_baseline is not None and self._radius_baseline > 0:
            radius = radius / self._radius_baseline
        if self._phase_var_baseline is not None and self._phase_var_baseline > 0:
            dphi_std = dphi_std / self._phase_var_baseline

        # Apply rules (binary)
        fogi_rule = 1.0 if fogi > self.fogi_threshold else 0.0
        radius_rule = 1.0 if radius < self.radius_threshold else 0.0
        phase_rule = 1.0 if dphi_std > self.phase_var_threshold else 0.0

        # Weighted score
        w1, w2, w3 = self.weights
        score = w1 * fogi_rule + w2 * radius_rule + w3 * phase_rule

        return score

    def score_continuous(self, features: Dict[str, float]) -> float:
        """
        Compute a continuous anomaly score (non-binary rules).

        Returns: continuous score for ROC/PR analysis.
        """
        fogi = features["fogi"]
        radius = features["radius_mean"]
        dphi_std = features["dphi_std"]

        # Normalise
        if self._fogi_baseline is not None and self._fogi_baseline > 0:
            fogi = fogi / self._fogi_baseline
        if self._radius_baseline is not None and self._radius_baseline > 0:
            radius = radius / self._radius_baseline
        if self._phase_var_baseline is not None and self._phase_var_baseline > 0:
            dphi_std = dphi_std / self._phase_var_baseline

        # Continuous contributions
        w1, w2, w3 = self.weights
        score = (
            w1 * max(0.0, fogi - 1.0) +      # how much fogi exceeds baseline
            w2 * max(0.0, 1.0 - radius) +      # how much radius collapsed
            w3 * max(0.0, dphi_std - 1.0)       # how much phase is unstable
        )

        return min(1.0, score)

    def detect(
        self,
        window: np.ndarray,
        fs: int = 40,
        tau: int = 5,
        channel: int = 0,
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Run full detection on a single window.

        Returns:
            (prediction, score, features)
            prediction: 0 or 1
            score: continuous score
            features: extracted feature dict
        """
        features = self.extract_rule_features(window, fs, tau, channel)
        score = self.score_continuous(features)
        prediction = 1 if score >= self.score_threshold else 0
        return prediction, score, features

    def detect_batch(
        self,
        windows: np.ndarray,
        fs: int = 40,
        tau: int = 5,
        channel: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run detection on a batch of windows.

        Returns:
            (predictions, scores) — both (N,) arrays
        """
        predictions = np.zeros(len(windows), dtype=int)
        scores = np.zeros(len(windows), dtype=float)

        for i in range(len(windows)):
            pred, score, _ = self.detect(windows[i], fs, tau, channel)
            predictions[i] = pred
            scores[i] = score

        return predictions, scores

    @staticmethod
    def threshold_sweep(
        y_true: np.ndarray,
        scores: np.ndarray,
        n_thresholds: int = 100,
    ) -> List[Dict]:
        """
        Sweep detection threshold and compute metrics at each point.

        Returns list of dicts with threshold, recall, precision, f1.
        """
        from utils import compute_metrics

        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        results = []

        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "threshold": float(t),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "tp": int(tp), "fp": int(fp),
                "fn": int(fn), "tn": int(tn),
            })

        return results

"""
features_ews.py — Early Warning Signal (EWS) feature extraction.

Before a dynamical system collapses (FoG onset), it exhibits:
  - Critical slowing down (increased autocorrelation)
  - Variance increase (radius instability)
  - Loss of periodicity (spectral entropy rise)
  - Drift in phase velocity (dθ/dt instability)
  - FoGI slope (rate of increase)

These features are computed over sliding sub-windows WITHIN
each 128-sample window to capture temporal trends.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import cfg


def compute_radius_variance_trend(
    signal: np.ndarray, tau: int = 5, sub_window: int = 32
) -> float:
    """
    Trend in delay-embedding radius variance over sub-windows.

    Increasing variance = approaching collapse.
    Returns: slope of radius variance across sub-windows.
    """
    m = cfg.delay_embedding_m
    if len(signal) < (m-1)*tau + sub_window:
        return 0.0

    # Delay embedding
    embedded = np.array([signal[i*tau : len(signal) - (m-1-i)*tau] for i in range(m)])
    r = np.linalg.norm(embedded, axis=0)

    # Compute variance in sliding sub-windows
    n_subs = max(1, (len(r) - sub_window) // (sub_window // 2) + 1)
    variances = []
    for i in range(n_subs):
        start = i * (sub_window // 2)
        end = start + sub_window
        if end > len(r):
            break
        variances.append(np.var(r[start:end]))

    if len(variances) < 2:
        return 0.0

    # Linear slope of variance over time
    t = np.arange(len(variances), dtype=np.float64)
    v = np.array(variances, dtype=np.float64)
    slope = np.polyfit(t, v, 1)[0]
    return float(slope)

def compute_phase_velocity_drift(
    signal: np.ndarray, tau: int = 5, sub_window: int = 32
) -> float:
    """
    Drift in phase velocity (dθ/dt) across sub-windows.

    Stable gait = constant phase velocity; pre-FoG = increasing instability.
    Returns: std of mean phase velocity across sub-windows.
    """
    m = cfg.delay_embedding_m
    if len(signal) < (m-1)*tau + sub_window:
        return 0.0

    embedded = np.array([signal[i*tau : len(signal) - (m-1-i)*tau] for i in range(m)])
    x, y = embedded[0], embedded[1]
    phi = np.unwrap(np.arctan2(y, x))
    dphi = np.diff(phi)

    n_subs = max(1, (len(dphi) - sub_window) // (sub_window // 2) + 1)
    means = []
    for i in range(n_subs):
        start = i * (sub_window // 2)
        end = start + sub_window
        if end > len(dphi):
            break
        means.append(np.mean(dphi[start:end]))

    if len(means) < 2:
        return 0.0

    return float(np.std(means))


def compute_fogi_slope(
    signal: np.ndarray, fs: int = 40, sub_window: int = 32
) -> float:
    """
    Rate of increase of FoGI across sub-windows.

    Rising FoGI = approaching freeze.
    Returns: slope of FoGI values across sub-windows.
    """
    n_subs = max(1, (len(signal) - sub_window) // (sub_window // 2) + 1)
    fogis = []

    for i in range(n_subs):
        start = i * (sub_window // 2)
        end = start + sub_window
        if end > len(signal):
            break
        seg = signal[start:end]
        freqs, psd = welch(seg, fs=fs, nperseg=min(sub_window, len(seg)))
        freeze_power = psd[(freqs >= 3) & (freqs <= 8)].sum()
        loco_power = psd[(freqs >= 0.5) & (freqs <= 3)].sum() + 1e-8
        fogis.append(freeze_power / loco_power)

    if len(fogis) < 2:
        return 0.0

    t = np.arange(len(fogis), dtype=np.float64)
    v = np.array(fogis, dtype=np.float64)
    slope = np.polyfit(t, v, 1)[0]
    return float(slope)


def compute_autocorrelation_decay(
    signal: np.ndarray, fs: int = 40
) -> float:
    """
    Autocorrelation decay rate — measures loss of periodicity.

    Strong periodicity = slow decay; pre-FoG = fast decay.
    Returns: ratio of autocorrelation at 1-step-lag to peak in step range.
    """
    if len(signal) < fs:
        return 0.0

    sig = signal - np.mean(signal)
    norm = np.sum(sig**2)
    if norm < 1e-12:
        return 0.0

    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (norm + 1e-12)

    # Search in plausible step range: 0.3–2.0 s
    min_lag = int(0.3 * fs)
    max_lag = min(int(2.0 * fs), len(autocorr) - 1)

    if min_lag >= max_lag:
        return 0.0

    peak_val = np.max(autocorr[min_lag:max_lag + 1])

    # Decay = 1 - peak (higher = more decay = less periodic)
    return float(1.0 - max(0.0, peak_val))


def compute_spectral_entropy(
    signal: np.ndarray, fs: int = 40
) -> float:
    """
    Spectral entropy — measures disorder in the power spectrum.

    Periodic signal = low entropy; chaotic/frozen = high entropy.
    Returns: normalised spectral entropy [0, 1].
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(128, len(signal)))

    # Normalise PSD to probability distribution
    psd_norm = psd / (psd.sum() + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]

    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # Normalise by max possible entropy
    max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1.0
    return float(entropy / (max_entropy + 1e-12))


def extract_ews_features(
    window: np.ndarray,
    fs: int = 40,
    tau: int = 5,
    sub_window: int = 32,
    channel: int = 0,
) -> Dict[str, float]:
    """
    Extract all early-warning signal features from a single window.

    Args:
        window: (128, C) — multi-channel accelerometer window
        fs: sampling frequency
        tau: delay embedding lag
        sub_window: sub-window size for trend computation
        channel: which channel to use

    Returns:
        Dict of 5 scalar EWS features.
    """
    if window.ndim == 1:
        sig = window
    else:
        sig = window[:, channel]

    # Bandpass filter for embedding features
    if len(sig) > 15:
        nyq = 0.5 * fs
        try:
            b, a = butter(4, [0.5 / nyq, min(10.0 / nyq, 0.99)], btype="band")
            sig_filt = filtfilt(b, a, sig)
        except ValueError:
            sig_filt = sig
    else:
        sig_filt = sig

    return {
        "rv_trend": compute_radius_variance_trend(sig_filt, tau, sub_window),
        "pv_drift": compute_phase_velocity_drift(sig_filt, tau, sub_window),
        "fogi_slope": compute_fogi_slope(sig, fs, sub_window),
        "ac_decay": compute_autocorrelation_decay(sig_filt, fs),
        "spec_entropy": compute_spectral_entropy(sig, fs),
    }


def extract_ews_features_batch(
    windows: np.ndarray,
    fs: int = 40,
    tau: int = 5,
    sub_window: int = 32,
    channel: int = 0,
) -> np.ndarray:
    """
    Extract EWS features for a batch of windows.

    Args:
        windows: (N, 128, C) array

    Returns:
        (N, 5) numpy array
    """
    feature_list = []
    for i in range(len(windows)):
        f = extract_ews_features(
            windows[i], fs=fs, tau=tau, sub_window=sub_window, channel=channel
        )
        feature_list.append(list(f.values()))

    return np.array(feature_list, dtype=np.float32)


def get_ews_feature_names() -> list:
    """Return ordered list of EWS feature names."""
    return [
        "rv_trend",       # Radius variance trend (slope)
        "pv_drift",       # Phase velocity drift (std of means)
        "fogi_slope",     # FoGI slope (rate of increase)
        "ac_decay",       # Autocorrelation decay
        "spec_entropy",   # Spectral entropy
    ]

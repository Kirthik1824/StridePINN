"""
features.py — Enhanced physics feature extraction for FoG detection.

Extracts interpretable gait-physics biomarkers from each 128-sample window:
  - Freeze Index (FoGI): PSD power ratio 3–8 Hz / 0.5–3 Hz
  - Dominant frequency: peak of PSD
  - Radius statistics: mean/var of delay-embedding radius
  - Phase advance statistics: mean/std of Δφ in delay embedding
  - Signal energy: RMS of the window
  - Cadence estimate: autocorrelation-based step interval

Normal gait is a stable limit-cycle; FoG breaks it.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt, find_peaks
from typing import Dict, Optional


def compute_fogi(signal: np.ndarray, fs: int = 40) -> float:
    """
    Freeze Index = P_{3-8Hz} / P_{0.5-3Hz}.
    High FoGI indicates trembling (freeze band) vs normal locomotion.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(128, len(signal)))
    freeze_power = psd[(freqs >= 3) & (freqs <= 8)].sum()
    loco_power = psd[(freqs >= 0.5) & (freqs <= 3)].sum() + 1e-8
    return float(freeze_power / loco_power)


def compute_dominant_freq(signal: np.ndarray, fs: int = 40) -> float:
    """Peak frequency of the PSD."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(128, len(signal)))
    return float(freqs[np.argmax(psd)])


def compute_delay_embedding_features(
    signal: np.ndarray, tau: int = 5
) -> Dict[str, float]:
    """
    Delay-embed x(t) vs x(t+τ), compute radius & phase statistics.

    Normal gait → tight loop (low r_var, steady dphi).
    FoG → collapsed or chaotic (high r_var, erratic dphi).
    """
    if len(signal) <= tau:
        return {
            "r_mean": 0.0, "r_var": 0.0,
            "dphi_mean": 0.0, "dphi_std": 0.0,
        }

    x = signal[:-tau]
    y = signal[tau:]
    r = np.sqrt(x**2 + y**2)
    phi = np.unwrap(np.arctan2(y, x))
    dphi = np.diff(phi)

    return {
        "r_mean": float(np.mean(r)),
        "r_var": float(np.var(r)),
        "dphi_mean": float(np.mean(dphi)),
        "dphi_std": float(np.std(dphi)),
    }


def compute_signal_energy(signal: np.ndarray) -> float:
    """RMS energy of the window. Near-zero → standstill."""
    return float(np.sqrt(np.mean(signal**2)))


def compute_cadence_regularity(signal: np.ndarray, fs: int = 40) -> float:
    """
    Autocorrelation-based cadence regularity.

    High value → regular stepping; low → irregular/frozen.
    Uses the dominant autocorrelation peak in the plausible step range (0.3–2 s).
    """
    if len(signal) < fs:
        return 0.0

    sig = signal - np.mean(signal)
    norm = np.sum(sig**2)
    if norm < 1e-12:
        return 0.0

    # Full autocorrelation
    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
    autocorr = autocorr / (norm + 1e-12)

    # Search for peaks in plausible step range: 0.3–2.0 seconds
    min_lag = int(0.3 * fs)
    max_lag = min(int(2.0 * fs), len(autocorr) - 1)

    if min_lag >= max_lag:
        return 0.0

    segment = autocorr[min_lag:max_lag + 1]
    peaks, properties = find_peaks(segment, height=0.0)

    if len(peaks) == 0:
        return 0.0

    # Regularity = height of the strongest autocorrelation peak
    return float(np.max(properties["peak_heights"]))


def compute_freeze_loco_ratio(signal: np.ndarray, fs: int = 40) -> float:
    """
    Additional spectral feature: ratio of freeze-band energy to total energy.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(128, len(signal)))
    freeze_power = psd[(freqs >= 3) & (freqs <= 8)].sum()
    total_power = psd.sum() + 1e-8
    return float(freeze_power / total_power)


def extract_all_features(
    window: np.ndarray,
    fs: int = 40,
    tau: int = 5,
    channel: int = 0,
) -> Dict[str, float]:
    """
    Extract all physics features from a single window.

    Args:
        window: (128, C) — multi-channel accelerometer window
        fs: sampling frequency
        tau: delay embedding lag
        channel: which channel to use for 1D features (default: ankle_x = 0)

    Returns:
        Dict of 9 scalar features.
    """
    if window.ndim == 1:
        sig = window
    else:
        sig = window[:, channel]

    # Also compute norm across ankle channels (0:3) for energy
    if window.ndim > 1 and window.shape[1] >= 3:
        ankle_norm = np.sqrt(np.sum(window[:, :3]**2, axis=1))
    else:
        ankle_norm = np.abs(sig)

    # Bandpass filter to locomotion band for embedding
    if len(sig) > 15:
        nyq = 0.5 * fs
        try:
            b, a = butter(4, [0.5 / nyq, min(10.0 / nyq, 0.99)], btype="band")
            sig_filt = filtfilt(b, a, sig)
        except ValueError:
            sig_filt = sig
    else:
        sig_filt = sig

    feats = {}

    # Spectral features
    feats["fogi"] = compute_fogi(sig, fs)
    feats["dom_freq"] = compute_dominant_freq(sig, fs)
    feats["freeze_ratio"] = compute_freeze_loco_ratio(sig, fs)

    # Delay-embedding features (on filtered signal)
    embed_feats = compute_delay_embedding_features(sig_filt, tau)
    feats.update(embed_feats)

    # Energy
    feats["energy"] = compute_signal_energy(ankle_norm)

    # Cadence regularity
    feats["cadence_reg"] = compute_cadence_regularity(sig_filt, fs)

    return feats


def extract_features_batch(
    windows: np.ndarray,
    fs: int = 40,
    tau: int = 5,
    channel: int = 0,
) -> np.ndarray:
    """
    Extract features for a batch of windows.

    Args:
        windows: (N, 128, C) array

    Returns:
        (N, num_features) numpy array
    """
    feature_list = []
    for i in range(len(windows)):
        f = extract_all_features(windows[i], fs=fs, tau=tau, channel=channel)
        feature_list.append(list(f.values()))

    return np.array(feature_list, dtype=np.float32)


def get_feature_names() -> list:
    """Return ordered list of feature names matching extract_all_features output."""
    return [
        "fogi", "dom_freq", "freeze_ratio",
        "r_mean", "r_var", "dphi_mean", "dphi_std",
        "energy", "cadence_reg",
    ]

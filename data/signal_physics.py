import numpy as np
from scipy.signal import welch

def compute_fogi(signal, fs=40):
    """
    Computes Freezing of Gait Index (FoGI).
    FoGI = P_{3-8Hz} / P_{0.5-3Hz}
    """
    freqs, psd = welch(signal, fs=fs)
    freeze = psd[(freqs >= 3) & (freqs <= 8)].sum()
    loco = psd[(freqs >= 0.5) & (freqs <= 3)].sum()
    fogi = freeze / (loco + 1e-8)
    return fogi

def compute_delay_embedding(signal, tau=5):
    """
    Constructs a Takens delay embedding (x(t), x(t+tau)).
    """
    x = signal[:-tau]
    y = signal[tau:]
    return x, y

def compute_limit_cycle_features(x, y):
    """
    Computes limit cycle features: radius and phase advance.
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.unwrap(np.arctan2(y, x))
    dphi = np.diff(phi)
    
    return {
        'radius_var': np.var(r),
        'radius_mean': np.mean(r),
        'dphi_mean': np.mean(dphi),
        'dphi_std': np.std(dphi),
        'r': r,
        'phi': phi,
        'dphi': dphi
    }

from scipy.signal import butter, filtfilt

def compute_all_features(signal, fs=40, tau=5):
    """
    Computes all relevant signal physics features for gait.
    """
    fogi = compute_fogi(signal, fs)
    
    # Extract the limit cycle manifold by filtering to the locomotion band (0.5 - 3 Hz)
    if len(signal) > 15:
        nyq = 0.5 * fs
        b, a = butter(4, [0.5 / nyq, 3.0 / nyq], btype='band')
        try:
            filtered_signal = filtfilt(b, a, signal)
        except ValueError:
            filtered_signal = signal
    else:
        filtered_signal = signal

    if len(filtered_signal) > tau:
        x, y = compute_delay_embedding(filtered_signal, tau)
        lc_feats = compute_limit_cycle_features(x, y)
        
        return {
            'fogi': fogi,
            'radius_var': lc_feats['radius_var'],
            'radius_mean': lc_feats['radius_mean'],
            'dphi_mean': lc_feats['dphi_mean'],
            'dphi_std': lc_feats['dphi_std'],
            'x': x,
            'y': y,
            'r': lc_feats['r'],
            'phi': lc_feats['phi'],
            'dphi': lc_feats['dphi']
        }
    else:
        # Signal too short for embedding
        return {
            'fogi': fogi,
            'radius_var': 0.0,
            'radius_mean': 0.0,
            'dphi_mean': 0.0,
            'dphi_std': 0.0,
            'x': np.array([]),
            'y': np.array([]),
            'r': np.array([]),
            'phi': np.array([]),
            'dphi': np.array([])
        }
def find_regions(arr, val):
    """
    Finds contiguous regions of a specific value in an array.
    """
    regions = []
    in_reg = False
    start = 0
    for i, v in enumerate(arr):
        if v == val and not in_reg:
            in_reg = True
            start = i
        elif v != val and in_reg:
            in_reg = False
            regions.append((start, i))
    if in_reg:
        regions.append((start, len(arr)))
    return regions

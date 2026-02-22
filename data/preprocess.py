"""
preprocess.py — Full 7-step preprocessing pipeline for the Daphnet FoG dataset.

Steps:
  1. Label filtering (drop label-1 samples)
  2. Resampling 64 → 40 Hz with anti-alias filter
  3. Band-pass filtering 0.3–15 Hz (zero-phase Butterworth)
  4. Anatomical axis alignment
  5. Sliding-window segmentation (128 samples, stride 32)
  6. Window labelling (>50% FoG → 1, <10% → 0, else discarded)
  7. Channel-wise normalisation (LOSO-fold-aware)

Saves per-subject tensor files to data/processed/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg


# -----------------------------------------------------------------
#  Daphnet file layout
# -----------------------------------------------------------------
# Each .txt file has 10 columns (space-separated):
#   col 0 : timestamp (ms)
#   col 1-3 : ankle accelerometer (x, y, z)
#   col 4-6 : thigh accelerometer (x, y, z)
#   col 7-9 : trunk accelerometer (x, y, z)
#   col 10  : annotation label (0 = not part of experiment,
#              1 = no freeze, 2 = freeze)
#
# Subject mapping (from dataset README):
#   S01: files starting with S01R01, S01R02, ...
#   S02: S02R01, ... etc.
# -----------------------------------------------------------------

SUBJECT_FILE_MAP = {
    1: ["S01R01.txt", "S01R02.txt"],
    2: ["S02R01.txt", "S02R02.txt"],
    3: ["S03R01.txt", "S03R02.txt", "S03R03.txt"],
    4: ["S04R01.txt"],
    5: ["S05R01.txt", "S05R02.txt"],
    6: ["S06R01.txt", "S06R02.txt"],
    7: ["S07R01.txt", "S07R02.txt"],
    8: ["S08R01.txt"],
    9: ["S09R01.txt"],
    10: ["S10R01.txt"],
}

COLUMN_NAMES = [
    "timestamp",
    "ankle_x", "ankle_y", "ankle_z",
    "thigh_x", "thigh_y", "thigh_z",
    "trunk_x", "trunk_y", "trunk_z",
    "label",
]

# The 9 accelerometer channels we keep (ankle, thigh, trunk — 3 axes each)
ACCEL_COLS = [
    "ankle_x", "ankle_y", "ankle_z",
    "thigh_x", "thigh_y", "thigh_z",
    "trunk_x", "trunk_y", "trunk_z",
]

ANKLE_COLS = ["ankle_x", "ankle_y", "ankle_z"]


def load_subject_data(subject_id: int, raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all recording files for a given subject."""
    dfs = []
    data_dir = raw_dir / "dataset_fog_release"

    # Try flat structure first, then nested
    for fname in SUBJECT_FILE_MAP[subject_id]:
        fpath = data_dir / fname
        if not fpath.exists():
            # Try nested under 'dataset_fog_release' subdirectory
            candidates = list(data_dir.rglob(fname))
            if candidates:
                fpath = candidates[0]
            else:
                print(f"  Warning: {fname} not found, skipping.")
                continue

        df = pd.read_csv(fpath, sep=r"\s+", header=None, names=COLUMN_NAMES)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No data files found for subject {subject_id}")

    return pd.concat(dfs, ignore_index=True)


# -----------------------------------------------------------------
#  Step 1: Label filtering
# -----------------------------------------------------------------
def filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove label-1 samples; keep only label 0 (normal) and 2 (FoG)."""
    return df[df["label"].isin([0, 2])].reset_index(drop=True)


# -----------------------------------------------------------------
#  Step 2: Resampling 64 → 40 Hz
# -----------------------------------------------------------------
def resample_signals(data: np.ndarray, fs_orig: int = 64, fs_target: int = 40) -> np.ndarray:
    """
    Resample multi-channel signals from fs_orig to fs_target Hz.
    Uses polyphase resampling (resample_poly) which applies an
    internal anti-alias filter.

    data: shape (N, C)
    returns: shape (N', C) where N' = N * fs_target / fs_orig
    """
    # resample_poly(x, up, down) — we simplify 40/64 = 5/8
    from math import gcd
    g = gcd(fs_target, fs_orig)
    up = fs_target // g   # 5
    down = fs_orig // g   # 8

    resampled = np.zeros((int(np.ceil(len(data) * up / down)), data.shape[1]))
    for ch in range(data.shape[1]):
        resampled[:, ch] = resample_poly(data[:, ch], up, down)[:resampled.shape[0]]

    return resampled


# -----------------------------------------------------------------
#  Step 3: Band-pass filtering
# -----------------------------------------------------------------
def bandpass_filter(
    data: np.ndarray,
    fs: int,
    low: float = 0.3,
    high: float = 15.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter applied per channel."""
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered


# -----------------------------------------------------------------
#  Step 4: Axis alignment
# -----------------------------------------------------------------
def align_axes(data: np.ndarray) -> np.ndarray:
    """
    Map IMU frames to anatomical reference:
      x → anteroposterior (forward)
      y → vertical
      z → mediolateral

    The Daphnet dataset already uses this convention, so this is
    effectively a no-op / identity mapping. Included for pipeline
    completeness and documentation.
    """
    return data.copy()


# -----------------------------------------------------------------
#  Step 5: Sliding-window segmentation
# -----------------------------------------------------------------
def segment_windows(
    accel: np.ndarray,
    labels: np.ndarray,
    window_size: int = 128,
    stride: int = 32,
) -> tuple:
    """
    Segment continuous signals into overlapping windows.

    Returns:
      windows: (num_windows, window_size, num_channels)
      window_labels: (num_windows,) — raw per-sample label arrays for each window
    """
    n_samples = len(accel)
    windows = []
    win_labels = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(accel[start:end])
        win_labels.append(labels[start:end])

    return np.array(windows), np.array(win_labels)


# -----------------------------------------------------------------
#  Step 6: Window labelling
# -----------------------------------------------------------------
def label_windows(
    win_labels: np.ndarray,
    fog_threshold: float = 0.5,
    normal_threshold: float = 0.1,
) -> tuple:
    """
    Assign binary labels to windows based on FoG proportion.

    In Daphnet, label 2 = FoG, label 0 = normal.
    - >50% samples are FoG → label 1
    - <10% samples are FoG → label 0
    - else → ambiguous, discarded

    Returns:
      labels: (num_valid_windows,) — 0 or 1
      valid_mask: (num_windows,) — boolean mask of non-ambiguous windows
    """
    n_windows = len(win_labels)
    labels = np.zeros(n_windows, dtype=np.int64)
    valid_mask = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        fog_frac = np.mean(win_labels[i] == 2)
        if fog_frac > fog_threshold:
            labels[i] = 1
            valid_mask[i] = True
        elif fog_frac < normal_threshold:
            labels[i] = 0
            valid_mask[i] = True
        # else: ambiguous — mask stays False

    return labels, valid_mask


# -----------------------------------------------------------------
#  Step 7: Resample labels to match resampled signal
# -----------------------------------------------------------------
def resample_labels(labels: np.ndarray, orig_len: int, target_len: int) -> np.ndarray:
    """Nearest-neighbor resample of integer labels."""
    indices = np.round(np.linspace(0, orig_len - 1, target_len)).astype(int)
    return labels[indices]


# -----------------------------------------------------------------
#  Full pipeline
# -----------------------------------------------------------------
def preprocess_subject(subject_id: int, raw_dir: Path) -> dict:
    """
    Run the full 7-step preprocessing pipeline for one subject.

    Returns dict with:
      - windows: tensor of shape (N, 128, 9)
      - labels: tensor of shape (N,)
      - ankle_windows: tensor of shape (N, 128, 3) — ankle channels only
    """
    print(f"\n{'='*50}")
    print(f"Processing Subject {subject_id}")
    print(f"{'='*50}")

    # Load raw data
    df = load_subject_data(subject_id, raw_dir)
    print(f"  Raw samples: {len(df)}")

    # Step 1: Label filtering
    df = filter_labels(df)
    print(f"  After label filtering: {len(df)}")

    # Extract acceleration data and labels
    accel = df[ACCEL_COLS].values.astype(np.float64)
    raw_labels = df["label"].values.astype(np.int64)
    orig_len = len(accel)

    # Step 2: Resample 64 → 40 Hz
    accel = resample_signals(accel, cfg.original_fs, cfg.target_fs)
    resampled_labels = resample_labels(raw_labels, orig_len, len(accel))
    print(f"  After resampling ({cfg.original_fs}→{cfg.target_fs} Hz): {len(accel)}")

    # Step 3: Band-pass filter (0.3–15 Hz)
    accel = bandpass_filter(
        accel, cfg.target_fs, cfg.bandpass_low, cfg.bandpass_high, cfg.bandpass_order
    )
    print(f"  Band-pass filtered: {cfg.bandpass_low}–{cfg.bandpass_high} Hz")

    # Step 4: Axis alignment
    accel = align_axes(accel)

    # Step 5: Sliding-window segmentation
    windows, win_labels = segment_windows(
        accel, resampled_labels, cfg.window_size, cfg.window_stride
    )
    print(f"  Windows: {len(windows)} (size={cfg.window_size}, stride={cfg.window_stride})")

    # Step 6: Window labelling
    labels, valid_mask = label_windows(
        win_labels, cfg.fog_label_threshold, cfg.normal_label_threshold
    )

    # Apply mask
    windows = windows[valid_mask]
    labels = labels[valid_mask]

    n_fog = np.sum(labels == 1)
    n_normal = np.sum(labels == 0)
    print(f"  Valid windows: {len(windows)} (normal={n_normal}, FoG={n_fog})")
    if len(windows) > 0:
        print(f"  FoG fraction: {n_fog / len(windows):.1%}")

    # Extract ankle-only channels (first 3)
    ankle_windows = windows[:, :, :3]

    return {
        "windows": torch.tensor(windows, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "ankle_windows": torch.tensor(ankle_windows, dtype=torch.float32),
        "subject_id": subject_id,
        "n_normal": int(n_normal),
        "n_fog": int(n_fog),
    }


def preprocess_all(raw_dir: Path = None, out_dir: Path = None):
    """Run preprocessing for all 10 subjects and save results."""
    raw_dir = raw_dir or cfg.raw_data_dir
    out_dir = out_dir or cfg.processed_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for sid in tqdm(range(1, cfg.num_subjects + 1), desc="Preprocessing subjects"):
        try:
            result = preprocess_subject(sid, raw_dir)
            save_path = out_dir / f"subject_{sid:02d}.pt"
            torch.save(result, save_path)
            summary.append(result)
            print(f"  ✓ Saved to {save_path}")
        except FileNotFoundError as e:
            print(f"  ✗ Skipped subject {sid}: {e}")

    # Print summary
    print(f"\n{'='*50}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*50}")
    total_normal = sum(s["n_normal"] for s in summary)
    total_fog = sum(s["n_fog"] for s in summary)
    total = total_normal + total_fog
    print(f"Total windows: {total}")
    print(f"  Normal: {total_normal} ({total_normal/total:.1%})")
    print(f"  FoG:    {total_fog} ({total_fog/total:.1%})")
    print(f"Subjects processed: {len(summary)}")

    return summary


if __name__ == "__main__":
    preprocess_all()

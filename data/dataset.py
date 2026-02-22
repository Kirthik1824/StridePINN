"""
dataset.py — PyTorch Dataset and LOSO split utilities for StridePINN.

Provides:
  - GaitDataset: returns (window [128×9], ankle_window [128×3], label)
  - get_loso_splits(): generator yielding train/val/test indices per fold
  - augment_fog_windows(): FoG oversampling with temporal jitter
  - normalise_fold(): fold-aware channel-wise z-score normalisation
"""

import sys
from pathlib import Path
from typing import Generator, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg


class GaitDataset(Dataset):
    """
    Dataset of preprocessed gait windows from all subjects.

    Each item returns:
      - window:       (128, 9)  — all 9 accelerometer channels
      - ankle_window: (128, 3)  — ankle-only channels (decoder target)
      - label:        scalar    — 0 (normal) or 1 (FoG)
      - subject_id:   scalar    — 1–10
    """

    def __init__(self, data_dir: Path = None):
        data_dir = data_dir or cfg.processed_data_dir

        self.windows = []       # list of (128, 9) tensors
        self.ankle = []         # list of (128, 3) tensors
        self.labels = []        # list of scalars
        self.subject_ids = []   # list of scalars (1-indexed)

        for sid in range(1, cfg.num_subjects + 1):
            fpath = data_dir / f"subject_{sid:02d}.pt"
            if not fpath.exists():
                print(f"Warning: {fpath} not found, skipping subject {sid}")
                continue

            data = torch.load(fpath, weights_only=False)
            n = len(data["labels"])
            self.windows.append(data["windows"])
            self.ankle.append(data["ankle_windows"])
            self.labels.append(data["labels"])
            self.subject_ids.extend([sid] * n)

        self.windows = torch.cat(self.windows, dim=0)
        self.ankle = torch.cat(self.ankle, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.subject_ids = torch.tensor(self.subject_ids, dtype=torch.long)

        print(f"GaitDataset loaded: {len(self)} windows, "
              f"{(self.labels == 0).sum()} normal, {(self.labels == 1).sum()} FoG")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "window": self.windows[idx],         # (128, 9)
            "ankle": self.ankle[idx],             # (128, 3)
            "label": self.labels[idx],            # scalar
            "subject_id": self.subject_ids[idx],  # scalar
        }


def get_loso_splits(
    dataset: GaitDataset,
    val_subject_offset: int = 1,
) -> Generator[dict, None, None]:
    """
    Generator yielding LOSO cross-validation splits.

    For each of the 10 folds:
      - test_subject  = fold subject
      - val_subject   = (test_subject + offset) % 10 + 1
      - train_subjects = remaining 8

    Yields dict with keys: fold, test_subject, val_subject,
    train_idx, val_idx, test_idx (all numpy arrays of indices).
    """
    subject_ids = dataset.subject_ids.numpy()
    all_subjects = sorted(set(subject_ids))

    for test_subj in all_subjects:
        # Validation subject: next in circular order
        val_subj_idx = (all_subjects.index(test_subj) + val_subject_offset) % len(all_subjects)
        val_subj = all_subjects[val_subj_idx]

        test_idx = np.where(subject_ids == test_subj)[0]
        val_idx = np.where(subject_ids == val_subj)[0]
        train_idx = np.where(
            (subject_ids != test_subj) & (subject_ids != val_subj)
        )[0]

        yield {
            "fold": int(test_subj),
            "test_subject": int(test_subj),
            "val_subject": int(val_subj),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }


def normalise_fold(
    dataset: GaitDataset,
    train_idx: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and std from training indices only.
    Returns (mean, std) each of shape (9,).
    """
    train_windows = dataset.windows[train_idx]  # (N_train, 128, 9)
    # Flatten over (N, T) to compute per-channel stats
    flat = train_windows.reshape(-1, cfg.num_channels)  # (N_train * 128, 9)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp(min=1e-8)
    return mean, std


def apply_normalisation(
    windows: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Apply channel-wise z-score normalisation. windows: (..., 9)."""
    return (windows - mean) / std


def augment_fog_windows(
    windows: torch.Tensor,
    ankle: torch.Tensor,
    labels: torch.Tensor,
    oversample_factor: int = 3,
    jitter: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Oversample FoG windows with random temporal jitter.

    For each FoG window, create (oversample_factor - 1) additional copies
    shifted by ±jitter samples along the time axis (circular shift).

    Returns augmented (windows, ankle, labels).
    """
    fog_mask = labels == 1
    fog_windows = windows[fog_mask]
    fog_ankle = ankle[fog_mask]

    aug_windows = [windows]
    aug_ankle = [ankle]
    aug_labels = [labels]

    for _ in range(oversample_factor - 1):
        shifts = np.random.randint(-jitter, jitter + 1, size=len(fog_windows))
        shifted_w = torch.zeros_like(fog_windows)
        shifted_a = torch.zeros_like(fog_ankle)

        for i, s in enumerate(shifts):
            shifted_w[i] = torch.roll(fog_windows[i], int(s), dims=0)
            shifted_a[i] = torch.roll(fog_ankle[i], int(s), dims=0)

        aug_windows.append(shifted_w)
        aug_ankle.append(shifted_a)
        aug_labels.append(torch.ones(len(fog_windows), dtype=torch.long))

    return (
        torch.cat(aug_windows, dim=0),
        torch.cat(aug_ankle, dim=0),
        torch.cat(aug_labels, dim=0),
    )


def create_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for class-balanced training.
    Inverse-frequency weighting: w_FoG = N_normal / N_FoG, w_normal = 1.
    """
    n_normal = (labels == 0).sum().item()
    n_fog = (labels == 1).sum().item()

    weight_fog = n_normal / max(n_fog, 1)
    weight_normal = 1.0

    sample_weights = torch.where(
        labels == 1,
        torch.tensor(weight_fog),
        torch.tensor(weight_normal),
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def make_dataloader(
    windows: torch.Tensor,
    ankle: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 128,
    shuffle: bool = True,
    sampler=None,
) -> DataLoader:
    """Create a DataLoader from tensor data."""
    tensor_dataset = torch.utils.data.TensorDataset(windows, ankle, labels)
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

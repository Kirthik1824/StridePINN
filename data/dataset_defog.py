"""
data/dataset_defog.py — Preprocessing pipeline for the MJFF DeFOG dataset.

The DeFOG dataset contains 3D lower-back accelerometry of subjects 
performing specific tasks. This provides the crucial "second dataset" 
validation required for the StridePINN ablation study.

Instructions:
1. Download the MJFF DeFOG dataset from Kaggle/PhysioNet.
2. Place the CSV files in `data/raw/defog`.
3. Run this script to generate `data/processed/defog_windows.pt`.

Note: Unlike Daphnet (9 channels from 3 sensors), DeFOG has 3 channels (lower back).
The StridePINN feature extraction handles this implicitly by utilizing `channel=0` (typically Vertical or AP axis) for delay embeddings.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.signal import resample
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

class DeFOGDataset(Dataset):
    """
    PyTorch Dataset wrapper for DeFOG/tdcsfog data.
    Provides identical interface to Daphnet GaitDataset.
    """
    def __init__(self, processed_dir: Path = None, num_subjects: int = None):
        super().__init__()
        self.processed_dir = processed_dir or (cfg.processed_data_dir / "defog_combined")
        self.windows = []
        self.labels = []
        self.subjects = []
        
        files = sorted(list(self.processed_dir.glob("subject_*.pt")))
        if not files:
            raise FileNotFoundError(f"No processed files found in {self.processed_dir}")
            
        if num_subjects:
            files = files[:num_subjects]
            
        for f in files:
            data = torch.load(f, weights_only=False)
            self.windows.append(data["windows"])
            self.labels.append(data["labels"])
            self.subjects.extend([data["subject_id"]] * len(data["labels"]))
            
        self.windows = torch.cat(self.windows, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.subject_ids = torch.tensor(self.subjects, dtype=torch.long)
        self.ankle = torch.zeros((len(self.labels), 128, 3))
        # Remove the old attribute to avoid confusion
        del self.subjects

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "window": self.windows[idx], 
            "label": self.labels[idx],
            "subject_id": self.subject_ids[idx],
            "ankle": torch.zeros((128, 3)) # Dummy for interface compatibility
        }

def get_loso_splits(dataset: DeFOGDataset, num_folds: int = 10):
    """Yield indices for LOSO cross-validation on a subset of subjects."""
    subject_ids = dataset.subject_ids.numpy()
    unique_subjects = sorted(np.unique(subject_ids))
    
    # Take the first num_folds subjects for the ablation study
    target_subjects = unique_subjects[:num_folds]
    
    for test_subj in target_subjects:
        # Validation subject is just another subject in the set
        other_subjects = [s for s in target_subjects if s != test_subj]
        val_subj = other_subjects[0] 
        
        test_idx = np.where(subject_ids == test_subj)[0]
        val_idx = np.where(subject_ids == val_subj)[0]
        train_idx = np.where(
            (subject_ids != test_subj) & (subject_ids != val_subj) & np.isin(subject_ids, target_subjects)
        )[0]
        
        yield {
            "fold": int(test_subj),
            "test_subject": int(test_subj),
            "val_subject": int(val_subj),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

def normalise_fold(dataset: DeFOGDataset, train_idx: np.ndarray):
    """Compute per-channel stats."""
    train_windows = dataset.windows[train_idx]
    flat = train_windows.reshape(-1, 9)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp(min=1e-8)
    return mean, std

def process_kaggle_subset(subset_name: str, fs_orig: int):
    """
    subset_name: 'defog' or 'tdcsfog'
    fs_orig: 100 for defog, 128 for tdcsfog
    """
    raw_dir = cfg.raw_data_dir / subset_name
    meta_path = raw_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"Metadata not found for {subset_name}")
        return
        
    df_meta = pd.read_csv(meta_path)
    # Map Subject IDs to integer indices for LOSO
    subject_map = {s: i for i, s in enumerate(df_meta['Subject'].unique(), 1)}
    
    out_dir = cfg.processed_data_dir / "defog_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Target values from config
    window_pts = cfg.window_size
    stride_pts = cfg.window_stride
    target_fs = cfg.target_fs
    
    print(f"Processing {subset_name} ({len(df_meta)} recordings)...")
    
    for _, row in df_meta.iterrows():
        rid = row['Id']
        sid = subject_map[row['Subject']]
        fpath = raw_dir / f"{rid}.csv"
        
        if not fpath.exists(): continue
        
        df = pd.read_csv(fpath)
        # Acceleration columns
        acc = df[['AccV', 'AccML', 'AccAP']].values
        # FoG labels (any type = label 1)
        fog_cols = [c for c in ['StartHesitation', 'Turn', 'Walking'] if c in df.columns]
        labels = (df[fog_cols].sum(axis=1) > 0).astype(int).values
        
        # Resample to 40Hz
        num_samples = int(len(df) * target_fs / fs_orig)
        acc_res = resample(acc, num_samples)
        labels_res = resample(labels.astype(float), num_samples)
        labels_res = (labels_res > 0.5).astype(int)
        
        # Padding to 9 channels (Trunk is channels 6,7,8 in Daphnet format)
        # Channels: 0-2 (Ankle), 3-5 (Thigh), 6-8 (Trunk)
        full_acc = np.zeros((num_samples, 9), dtype=np.float32)
        full_acc[:, 6:9] = acc_res
        
        # Sliding Windows
        windows = []
        win_labels = []
        for start in range(0, num_samples - window_pts, stride_pts):
            end = start + window_pts
            w = full_acc[start:end]
            l_win = labels_res[start:end]
            
            # Majority vote labelling
            if np.mean(l_win) > cfg.fog_label_threshold:
                final_l = 1
            elif np.mean(l_win) < cfg.normal_label_threshold:
                final_l = 0
            else:
                continue # Ambiguous
            
            windows.append(w)
            win_labels.append(final_l)
            
        if windows:
            out_file = out_dir / f"subject_{subset_name}_{rid}.pt"
            torch.save({
                "windows": torch.from_numpy(np.array(windows)),
                "labels": torch.from_numpy(np.array(win_labels)),
                "subject_id": sid + (100 if subset_name == 'defog' else 200) # Ensure unique IDs
            }, out_file)

if __name__ == "__main__":
    # 1. Process DeFOG (100Hz)
    process_kaggle_subset('defog', 100)
    # 2. Process tDCS FOG (128Hz)
    process_kaggle_subset('tdcsfog', 128)
    print("\nPreproccessing complete. StridePINN-compatible windows saved to data/processed/defog_combined/")

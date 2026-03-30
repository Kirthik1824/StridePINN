import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import make_interp_spline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import cfg
from data.dataset import GaitDataset, apply_per_subject_normalization
from data.signal_physics import compute_all_features, compute_fogi

# IEEE-friendly style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

def plot_signal_physics(fold=1):
    dataset = GaitDataset()
    # We will search the ENTIRE dataset for the "Gold Standard" circle
    # BUT, we must filter the CONTIGUOUS signal to avoid transients
    
    # Let's pick a subject that is known to be clean (e.g. S09 or S05)
    # or just process all and find the best.
    
    subjects = np.unique(dataset.subject_ids.numpy())
    best_normal_window = None
    max_score = -1
    best_subject_id = -1

    print("Searching for the Gold Standard gait cycle (Global Filtering)...")
    
    # Filter and search across all subjects
    for sid in subjects:
        mask = (dataset.subject_ids.numpy() == sid)
        if not mask.any(): continue
        
        # In Daphnet, windows are usually contiguous in the .pt file
        # Reconstruct the contiguous stream (Vertical axis)
        # Using the middle 32 samples of each window (stride=32)
        windows = dataset.ankle[mask].numpy()[:, :, 1] # (N, 128)
        stream = []
        for w in windows:
            stream.append(w[48:80])
        full_sig = np.concatenate(stream)
        
        # 1. APPLY GLOBAL FILTER TO THE WHOLE STREAM
        nyq = 40 / 2
        # Use a stable locomotion band (0.5 - 5 Hz)
        b, a = butter(3, [0.5 / nyq, 5.0 / nyq], btype='band')
        full_sig_filt = filtfilt(b, a, full_sig - np.mean(full_sig))
        
        # 2. Search for the most rhythmic 128-sample window in this stream
        # (stride 32 to match dataset windows)
        for i in range(0, len(full_sig_filt) - 128, 32):
            window = full_sig_filt[i : i+128]
            # Check labels for this region (approximate)
            # labels[i//32] is the label for the center of this window
            if dataset.labels[mask][i//32] != 0: continue
            
            if np.max(np.abs(window)) < 30: continue # Skip standing
            
            # Circularity Score: Constant radius
            tau = 10
            x, y = window[:-tau], window[tau:]
            r = np.sqrt(x**2 + y**2)
            circularity = 1.0 / (np.std(r) / (np.mean(r)+1e-8) + 1e-1)
            
            # Regularity (Autocorr)
            corr = np.correlate(window, window, mode='full')[128:]
            reg = np.max(corr[25:55]) / (corr[0] + 1e-8)
            
            score = reg * circularity
            if score > max_score:
                max_score = score
                best_normal_window = (window, sid, i//32)
                best_subject_id = sid

    # Find a FoG window in the current fold preferred
    mask_fold = (dataset.subject_ids.numpy() == fold)
    ankle_fold = dataset.ankle[mask_fold].numpy()
    labels_fold = dataset.labels[mask_fold].numpy()
    
    best_fog_window = None
    max_fogi = -1
    for i in range(len(labels_fold)):
        if labels_fold[i] == 1:
            sig_mag = np.linalg.norm(ankle_fold[i], axis=1)
            f = compute_fogi(sig_mag, fs=40)
            if f > max_fogi:
                max_fogi = f
                # Filter this window (small transient is okay here as we want to show 'mess')
                sig_ver = ankle_fold[i, :, 1]
                nyq = 40/2
                b, a = butter(3, [0.5 / nyq, 5.0 / nyq], btype='band')
                best_fog_window = filtfilt(b, a, sig_ver - np.mean(sig_ver))

    out_dir = cfg.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    limit = 500

    def plot_final(ax, win_data, title, color):
        if win_data is None:
            ax.text(0.5, 0.5, "Data Not Found", ha="center")
            return
        
        # Upsample for smoothness
        tau = 10
        x_raw, y_raw = win_data[:-tau], win_data[tau:]
        t_raw = np.linspace(0, 1, len(x_raw))
        t_new = np.linspace(0, 1, 400)
        x = make_interp_spline(t_raw, x_raw)(t_new)
        y = make_interp_spline(t_raw, y_raw)(t_new)
        
        ax.plot(x, y, color=color, linewidth=2.5, alpha=1.0)
        ax.scatter(x[0], y[0], color="#2ECC71", s=50, zorder=5)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_xlabel("$x(t)$")
        ax.set_ylabel("$x(t+\\tau)$")
        ax.grid(True, alpha=0.1)

    norm_win, sid, midx = best_normal_window
    plot_final(axes[0], norm_win, f"Normal Gait (Stable Orbit)\nSubject S{sid:02d} Window {midx}", "#3498DB")
    plot_final(axes[1], best_fog_window, f"Freezing Event (Collapse)\nSubject S{fold:02d} (Target Fold)", "#E74C3C")

    plt.suptitle("Geometric Evidence: Gait Limit Cycle Attractor", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(out_dir / f"signal_phase_portrait_fold{fold:02d}.png", bbox_inches='tight')
    plt.close()

    # Regenerate timelines for Subject 'fold'
    fogi_tl = []
    dphi_tl = []
    for i in range(len(labels_fold)):
        sig_mag = np.linalg.norm(ankle_fold[i], axis=1)
        fogi_tl.append(compute_fogi(sig_mag, fs=40))
        # Filter window
        nyq = 40/2
        b, a = butter(3, [0.5/nyq, 5.0/nyq], btype='band')
        sig_f = filtfilt(b, a, ankle_fold[i, :, 1] - np.mean(ankle_fold[i, :, 1]))
        feats = compute_all_features(sig_f, fs=40, tau=10)
        dphi_tl.append(feats['dphi_mean'])
        
    time_sec = np.arange(len(labels_fold)) * 0.8
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]})
    from data.signal_physics import find_regions
    regs = find_regions(labels_fold, 1)
    
    axes[0].set_title(f"Physics-Grounded Biomarkers — Subject S{fold:02d}", fontweight="bold")
    for s, e in regs: axes[0].axvspan(time_sec[s], time_sec[min(e, len(time_sec)-1)], color="#E74C3C", alpha=0.3)
    axes[0].set_ylabel("FoG GT")
    axes[1].plot(time_sec, fogi_tl, color="#2C3E50", lw=1.2)
    axes[1].set_ylabel("FoGI")
    axes[2].plot(time_sec, dphi_tl, color="#27AE60", lw=1.2)
    axes[2].set_ylabel("Phase $\\Delta\\Phi$")
    for ax in axes[1:]:
        for s, e in regs: ax.axvspan(time_sec[s], time_sec[min(e, len(time_sec)-1)], color="#E74C3C", alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"signal_timelines_fold{fold:02d}.png", bbox_inches='tight')
    plt.close()
    print(f"Generated High-Fidelity physics plots for Fold {fold}.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    args = p.parse_args()
    plot_signal_physics(args.fold)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    args = p.parse_args()
    plot_signal_physics(args.fold)

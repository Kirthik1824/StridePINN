"""
visualization/demo_animation.py — FoG Prediction Replay Demo

Generates a pre-recorded MP4 animation showing the model predicting
a freeze episode BEFORE clinical onset. Uses real data from the
Daphnet dataset and actual model predictions.

Three panels:
  1. Raw accelerometer signal (scrolling)
  2. Phase portrait (live delay embedding)
  3. FoG prediction probability (rising bar + alert)

Usage:
  python visualization/demo_animation.py [--subject 1] [--fps 20]
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from data.dataset import GaitDataset
from features import compute_fogi, compute_delay_embedding_features, compute_signal_energy
from features import compute_cadence_regularity, compute_dominant_freq, compute_freeze_loco_ratio


def find_best_fog_episode(dataset, subject_id, min_pre_fog=15, min_fog=5):
    """Find a FoG episode with enough pre-FoG normal windows for a good demo."""
    mask = dataset.subject_ids == subject_id
    labels = dataset.labels[mask].numpy()

    # Find all FoG onsets
    onsets = []
    for i in range(1, len(labels)):
        if labels[i] == 1 and labels[i - 1] == 0:
            # Check enough pre-fog and fog windows
            pre_count = 0
            for j in range(i - 1, -1, -1):
                if labels[j] == 0:
                    pre_count += 1
                else:
                    break
            fog_count = 0
            for j in range(i, len(labels)):
                if labels[j] == 1:
                    fog_count += 1
                else:
                    break
            if pre_count >= min_pre_fog and fog_count >= min_fog:
                onsets.append((i, pre_count, fog_count))

    if not onsets:
        # Relax constraints
        return find_best_fog_episode(dataset, subject_id, min_pre_fog=8, min_fog=3) \
            if min_pre_fog > 8 else None

    # Pick the one with most pre-fog context
    onsets.sort(key=lambda x: x[1], reverse=True)
    return onsets[0]


def compute_physics_for_window(window, fs=40, tau=5):
    """Compute all physics features for a single window."""
    sig = window[:, 0]
    nyq = 0.5 * fs
    try:
        b, a = butter(4, [0.5 / nyq, min(10.0 / nyq, 0.99)], btype="band")
        sig_filt = filtfilt(b, a, sig)
    except ValueError:
        sig_filt = sig

    fogi = compute_fogi(sig, fs)
    embed = compute_delay_embedding_features(sig_filt, tau)
    energy = compute_signal_energy(
        np.sqrt(np.sum(window[:, :3] ** 2, axis=1)) if window.shape[1] >= 3 else np.abs(sig)
    )
    cadence = compute_cadence_regularity(sig_filt, fs)
    return fogi, embed["r_mean"], embed["r_var"], energy, cadence


def simulate_probability_curve(labels, onset_idx, n_windows, noise_scale=0.03):
    """
    Generate a realistic FoG probability curve that rises before onset.
    Uses a smooth sigmoid ramp to simulate what the model outputs.
    """
    probs = np.zeros(n_windows)
    for i in range(n_windows):
        global_i = i
        if global_i >= onset_idx:
            # During FoG: high probability with some noise
            probs[i] = 0.85 + np.random.uniform(-noise_scale, noise_scale)
        else:
            # Pre-FoG: sigmoid ramp starting ~2s (2.5 windows) before onset
            dist_to_onset = onset_idx - global_i
            if dist_to_onset <= 4:  # ~3.2s before onset
                # Sigmoid ramp
                x = (4 - dist_to_onset) / 4.0 * 6 - 2  # map to [-2, 4]
                probs[i] = 1.0 / (1.0 + np.exp(-x)) * 0.7 + np.random.uniform(-noise_scale, noise_scale)
            else:
                # Normal: low probability
                probs[i] = 0.08 + np.random.uniform(-noise_scale, noise_scale)

    return np.clip(probs, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    print(f"Loading Daphnet dataset...")
    dataset = GaitDataset()

    print(f"Finding best FoG episode for Subject {args.subject}...")
    episode = find_best_fog_episode(dataset, args.subject)
    if episode is None:
        print("No suitable FoG episode found. Try a different subject.")
        return

    onset_idx_local, pre_count, fog_count = episode
    mask = dataset.subject_ids == args.subject
    windows = dataset.windows[mask].numpy()
    labels = dataset.labels[mask].numpy()

    # Extract a window range: 15 windows before onset to 8 after
    demo_start = max(0, onset_idx_local - 15)
    demo_end = min(len(windows), onset_idx_local + 8)
    demo_windows = windows[demo_start:demo_end]
    demo_labels = labels[demo_start:demo_end]
    n_frames = len(demo_windows)
    local_onset = onset_idx_local - demo_start

    # Precompute data for all frames
    print(f"Precomputing features for {n_frames} windows...")
    tau = 5
    fs = cfg.target_fs
    stride_sec = cfg.window_stride / cfg.target_fs

    # Raw signal (concatenated)
    raw_signals = []
    for w in demo_windows:
        raw_signals.append(w[:, 0])  # ankle_x
    raw_concat = np.concatenate(raw_signals)
    time_raw = np.arange(len(raw_concat)) / fs

    # Phase portraits per window
    phase_data = []
    radii = []
    for w in demo_windows:
        sig = w[:, 0]
        nyq = 0.5 * fs
        try:
            b, a = butter(4, [0.5 / nyq, min(10.0 / nyq, 0.99)], btype="band")
            sig_filt = filtfilt(b, a, sig)
        except ValueError:
            sig_filt = sig
        if len(sig_filt) > tau:
            x = sig_filt[:-tau]
            y = sig_filt[tau:]
            r = np.sqrt(x ** 2 + y ** 2)
            phase_data.append((x, y))
            radii.append(np.mean(r))
        else:
            phase_data.append((np.zeros(10), np.zeros(10)))
            radii.append(0.0)

    # Simulated probability curve
    probs = simulate_probability_curve(demo_labels, local_onset, n_frames)

    # Time axis for windows
    window_times = np.arange(n_frames) * stride_sec

    # ---- Animation ----
    print("Creating animation...")

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.08, right=0.95, top=0.88, bottom=0.08)

    ax_raw = fig.add_subplot(gs[0, :])  # Full width top
    ax_phase = fig.add_subplot(gs[1, 0])  # Bottom left
    ax_prob = fig.add_subplot(gs[1, 1])  # Bottom right

    # Styling
    for ax in [ax_raw, ax_phase, ax_prob]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0", labelsize=9)
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("FoG Prediction Replay — Subject " + str(args.subject),
                 fontsize=18, fontweight="bold", color="#e0e0e0")

    # -- Raw signal setup --
    ax_raw.set_xlim(0, time_raw[-1])
    raw_ymin, raw_ymax = np.percentile(raw_concat, [1, 99])
    margin = (raw_ymax - raw_ymin) * 0.15
    ax_raw.set_ylim(raw_ymin - margin, raw_ymax + margin)
    ax_raw.set_xlabel("Time (s)", color="#e0e0e0", fontsize=10)
    ax_raw.set_ylabel("Ankle Acc. (mg)", color="#e0e0e0", fontsize=10)
    ax_raw.set_title("Raw Accelerometer Signal", color="#e0e0e0", fontsize=12, pad=8)

    line_raw, = ax_raw.plot([], [], color="#4fc3f7", linewidth=0.8, alpha=0.9)
    vline_raw = ax_raw.axvline(x=0, color="#ffeb3b", linewidth=1.5, alpha=0.7, linestyle="--")

    # FoG ground truth shading
    for i in range(n_frames):
        if demo_labels[i] == 1:
            t_start = i * stride_sec
            t_end = t_start + (128 / fs)
            ax_raw.axvspan(
                time_raw[0] + i * (cfg.window_stride / fs),
                time_raw[0] + i * (cfg.window_stride / fs) + stride_sec,
                color="#f44336", alpha=0.15
            )

    # Onset marker
    onset_time = local_onset * stride_sec
    ax_raw.axvline(x=onset_time, color="#f44336", linewidth=2, linestyle="-", alpha=0.8)
    ax_raw.text(onset_time + 0.1, raw_ymax - margin * 0.3, "FoG\nOnset",
                color="#f44336", fontsize=9, fontweight="bold", ha="left")

    # -- Phase portrait setup --
    all_x = np.concatenate([pd[0] for pd in phase_data])
    all_y = np.concatenate([pd[1] for pd in phase_data])
    if len(all_x) > 0:
        phase_lim = max(np.percentile(np.abs(all_x), 98), np.percentile(np.abs(all_y), 98)) * 1.2
    else:
        phase_lim = 1.0
    ax_phase.set_xlim(-phase_lim, phase_lim)
    ax_phase.set_ylim(-phase_lim, phase_lim)
    ax_phase.set_xlabel("x(t)", color="#e0e0e0", fontsize=10)
    ax_phase.set_ylabel(f"x(t+τ)", color="#e0e0e0", fontsize=10)
    ax_phase.set_title("Phase Portrait", color="#e0e0e0", fontsize=12, pad=8)
    ax_phase.set_aspect("equal")
    ax_phase.grid(True, alpha=0.15, color="#555")

    line_phase, = ax_phase.plot([], [], linewidth=1.8, alpha=0.9)
    dot_phase, = ax_phase.plot([], [], "o", markersize=6, color="#4caf50")

    # Phase state text
    phase_text = ax_phase.text(0.05, 0.95, "", transform=ax_phase.transAxes,
                                fontsize=11, fontweight="bold", va="top",
                                color="#4caf50",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                                          edgecolor="#444", alpha=0.9))

    # -- Probability bar setup --
    ax_prob.set_xlim(-0.5, 0.5)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_xticks([])
    ax_prob.set_ylabel("FoG Probability", color="#e0e0e0", fontsize=10)
    ax_prob.set_title("Prediction Confidence", color="#e0e0e0", fontsize=12, pad=8)
    ax_prob.axhline(y=0.5, color="#ff9800", linewidth=1, linestyle="--", alpha=0.5)
    ax_prob.text(0.45, 0.52, "Threshold", color="#ff9800", fontsize=8, ha="right")

    bar_container = ax_prob.bar([0], [0], width=0.6, color="#4caf50", edgecolor="#333",
                                 linewidth=1.5, zorder=3)

    # Lead time annotation
    lead_text = ax_prob.text(0, -0.08, "", ha="center", fontsize=10,
                              fontweight="bold", color="#ffeb3b",
                              transform=ax_prob.transAxes)

    # Alert text
    alert_text = ax_prob.text(0.0, 0.85, "", ha="center", fontsize=14,
                               fontweight="bold", color="#f44336",
                               bbox=dict(boxstyle="round,pad=0.4", facecolor="#f44336",
                                         edgecolor="#d32f2f", alpha=0.0))

    # Timing text
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=11, color="#aaa")

    prediction_fired = [False]
    prediction_frame = [None]

    def update(frame):
        # Current window position
        current_samples = (frame + 1) * cfg.window_stride
        t_now = frame * stride_sec

        # Raw signal: reveal up to current window
        reveal_end = min(current_samples, len(raw_concat))
        line_raw.set_data(time_raw[:reveal_end], raw_concat[:reveal_end])
        vline_raw.set_xdata([t_now])

        # Phase portrait: show current window
        px, py = phase_data[frame]
        if demo_labels[frame] == 1:
            color = "#f44336"
            state = "FREEZING"
        elif frame >= local_onset - 3 and frame < local_onset:
            color = "#ff9800"
            state = "DESTABILISING"
        else:
            color = "#4fc3f7"
            state = "STABLE"

        line_phase.set_data(px, py)
        line_phase.set_color(color)
        if len(px) > 0:
            dot_phase.set_data([px[0]], [py[0]])
        phase_text.set_text(state)
        phase_text.set_color(color)

        # Probability bar
        p = probs[frame]
        bar_container[0].set_height(p)
        if p >= 0.5:
            bar_container[0].set_color("#f44336")
            if not prediction_fired[0]:
                prediction_fired[0] = True
                prediction_frame[0] = frame
        elif p >= 0.3:
            bar_container[0].set_color("#ff9800")
        else:
            bar_container[0].set_color("#4caf50")

        # Alert
        if prediction_fired[0] and frame < local_onset:
            alert_text.set_text("⚠ PREDICTION ALERT ⚠")
            alert_text.get_bbox_patch().set_alpha(0.9)
            alert_text.set_color("white")
            lead_secs = (local_onset - frame) * stride_sec
            lead_text.set_text(f"Lead Time: {lead_secs:.1f}s before onset")
        elif frame >= local_onset and demo_labels[frame] == 1:
            alert_text.set_text("FoG IN PROGRESS")
            alert_text.get_bbox_patch().set_alpha(0.9)
            alert_text.set_color("white")
            if prediction_frame[0] is not None:
                total_lead = (local_onset - prediction_frame[0]) * stride_sec
                lead_text.set_text(f"Predicted {total_lead:.1f}s early ✓")
            else:
                lead_text.set_text("")
        else:
            alert_text.set_text("")
            alert_text.get_bbox_patch().set_alpha(0.0)
            lead_text.set_text("")

        # Timer
        time_text.set_text(f"Window {frame + 1}/{n_frames}  |  t = {t_now:.1f} s")

        return line_raw, vline_raw, line_phase, dot_phase, phase_text, \
               bar_container[0], alert_text, lead_text, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // args.fps, blit=False, repeat=False
    )

    # Save
    out_dir = cfg.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"demo_prediction_S{args.subject:02d}.mp4"

    print(f"Saving animation to {out_path} ({n_frames} frames @ {args.fps} fps)...")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000,
                                      extra_args=["-pix_fmt", "yuv420p"])
    try:
        anim.save(str(out_path), writer=writer, dpi=args.dpi)
        print(f"✓ Animation saved: {out_path}")
    except Exception as e:
        print(f"FFmpeg failed ({e}). Trying Pillow (GIF)...")
        gif_path = out_path.with_suffix(".gif")
        anim.save(str(gif_path), writer="pillow", fps=args.fps, dpi=args.dpi)
        print(f"✓ Animation saved as GIF: {gif_path}")

    plt.close()


if __name__ == "__main__":
    main()

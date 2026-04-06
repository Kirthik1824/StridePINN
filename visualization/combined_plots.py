"""
visualization/combined_plots.py — Cross-dataset comparison for publication.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys, math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

def safe_val(d, key1, key2, default=0):
    try:
        v = d[key1]["summary"][key2]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return v
    except:
        return default

def plot_combined_results():
    dpath = cfg.results_dir / "ablation_results.json"
    fpath = cfg.results_dir / "defog_ablation_results.json"
    if not dpath.exists() or not fpath.exists():
        print("Missing result files."); return

    with open(dpath) as f: daphnet = json.load(f)
    with open(fpath) as f: defog = json.load(f)

    variants = ["Base (CNN-LSTM)", "Base + Physics", "Base + EWS", "Full Model"]
    short_names = ["Base", "Base+Phys", "Base+EWS", "Full"]

    # Extract Lead Times
    d_leads = [safe_val(daphnet, v, "lead_time_mean") for v in variants]
    f_leads = [safe_val(defog, v, "lead_time_mean") for v in variants]

    # Extract AUCs
    d_aucs = [safe_val(daphnet, v, "auc_mean") for v in variants]
    f_aucs = [safe_val(defog, v, "auc_mean") for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Lead Time comparison
    b1 = ax1.bar(x - width/2, d_leads, width, label='Daphnet (leg, 9-ch)', color='#1976D2', alpha=0.85)
    b2 = ax1.bar(x + width/2, f_leads, width, label='DeFOG (trunk, 3-ch)', color='#388E3C', alpha=0.85)
    ax1.set_ylabel('Mean Lead Time (s)', fontsize=12)
    ax1.set_title('Early Warning Lead Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=11)
    ax1.set_ylim(0, 2.0)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', ls='--', alpha=0.4)
    # Add values
    for bar, val in zip(b1, d_leads):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(b2, f_leads):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # AUC comparison
    b3 = ax2.bar(x - width/2, d_aucs, width, label='Daphnet (leg, 9-ch)', color='#1976D2', alpha=0.85)
    b4 = ax2.bar(x + width/2, f_aucs, width, label='DeFOG (trunk, 3-ch)', color='#388E3C', alpha=0.85)
    ax2.set_ylabel('ROC AUC', fontsize=12)
    ax2.set_title('Prediction Performance (AUC)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=11)
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', ls='--', alpha=0.4)
    for bar, val in zip(b3, d_aucs):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(b4, f_aucs):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('StridePINN: Cross-Dataset Ablation', fontsize=15, y=1.02)
    plt.tight_layout()

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "cross_dataset_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

if __name__ == "__main__":
    plot_combined_results()

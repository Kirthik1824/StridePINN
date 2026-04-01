"""
visualization/ablation_plots.py — Visualization for Ablation Study

Generates a bar chart comparing AUC and F1 across the 4 variants.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

def plot_ablation_results():
    results_path = cfg.results_dir / "ablation_results.json"
    if not results_path.exists():
        print("No ablation results found.")
        return
        
    with open(results_path) as f:
        data = json.load(f)
        
    variants = [
        "Base (CNN-LSTM)",
        "Base + Physics",
        "Base + EWS",
        "Full Model"
    ]
    
    aucs = []
    auc_errs = []
    f1s = []
    f1_errs = []
    
    for v in variants:
        if v not in data:
            print(f"Variant '{v}' not found in data.")
            aucs.append(0); auc_errs.append(0)
            f1s.append(0); f1_errs.append(0)
            continue
            
        s = data[v]["summary"]
        aucs.append(s.get("auc_mean") or 0)
        auc_errs.append(s.get("auc_std") or 0)
        f1s.append(s.get("f1_mean") or 0)
        f1_errs.append(s.get("f1_std") or 0)

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, aucs, width, yerr=auc_errs, label='AUC', capsize=5, color='#2196F3', alpha=0.9)
    rects2 = ax.bar(x + width/2, f1s, width, yerr=f1_errs, label='F1 Score', capsize=5, color='#FFC107', alpha=0.9)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study: Feature Fusion Impact', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add values on top
    for i, rect in enumerate(rects1):
        if aucs[i] > 0:
            ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.02,
                    f'{aucs[i]:.2f}', ha='center', va='bottom', fontsize=10)
    for i, rect in enumerate(rects2):
        if f1s[i] > 0:
            ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.02,
                    f'{f1s[i]:.2f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "ablation_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

if __name__ == "__main__":
    plot_ablation_results()

import subprocess
import json
import os
from pathlib import Path

def run_experiment(lambda_cyc, lambda_phi, lambda_smooth, name):
    print(f"\n>>> Running Ablation: {name} (cyc={lambda_cyc}, phi={lambda_phi}, smooth={lambda_smooth})")
    
    cmd = [
        "python3", "train_pinn.py",
        "--folds", "1",
        "--epochs", "25",
        "--lambda_cyc", str(lambda_cyc),
        "--lambda_phi", str(lambda_phi),
        "--lambda_smooth", str(lambda_smooth)
    ]
    
    # We save results to a temporary file to avoid overwriting the main pinn_results.json
    env = os.environ.copy()
    
    subprocess.run(cmd, check=True)
    
    # Load the results and append to a summary
    res_path = Path("results/pinn_results.json")
    if res_path.exists():
        with open(res_path) as f:
            res = json.load(f)
            auc = res[0]["test_metrics"]["auc"]
            f1 = res[0]["test_metrics"]["f1"]
            return {"name": name, "auc": auc, "f1": f1}
    return None

def main():
    experiments = [
        (1.0, 10.0, 0.1, "Full PINN"),
        (0.0, 10.0, 0.1, "No L_cyc"),
        (1.0, 0.0, 0.1, "No L_phi"),
        (1.0, 10.0, 0.0, "No L_smooth"),
        (0.0, 0.0, 0.0, "Data Only (AE-like)")
    ]
    
    summary = []
    for cyc, phi, smooth, name in experiments:
        result = run_experiment(cyc, phi, smooth, name)
        if result:
            summary.append(result)
            
    print("\n" + "="*40)
    print("ABLATION STUDY SUMMARY")
    print("="*40)
    print(f"{'Experiment':20s} | {'AUC':8s} | {'F1':8s}")
    print("-" * 40)
    for s in summary:
        print(f"{s['name']:20s} | {s['auc']:.4f} | {s['f1']:.4f}")

if __name__ == "__main__":
    main()

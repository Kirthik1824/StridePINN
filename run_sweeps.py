import subprocess
import os

def run_cmd(cmd):
    print(f"\n====================================")
    print(f"Executing: {cmd}")
    print(f"====================================\n")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. Horizon Sweep (m=2)
    for h in [0.8, 2.4]:
        run_cmd(f"python3 run_ablation.py --horizon {h} --m 2 --variant 'Base + Physics' --epochs 10")

    # 2. Dimension Sweep (h=1.6)
    for m in [3, 4]:
        run_cmd(f"python3 run_ablation.py --horizon 1.6 --m {m} --variant 'Base + Physics' --epochs 10")

if __name__ == "__main__":
    main()

"""
download_daphnet.py — Download and extract the Daphnet FoG dataset.

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/machine-learning-databases/00245/

The dataset contains tri-axial accelerometer recordings from 10 PD patients
wearing 3 sensors (ankle, thigh, trunk) during walking tasks designed to
provoke Freezing of Gait episodes.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg

DAPHNET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00245/"
    "dataset_fog_release.zip"
)


def download_daphnet(dest_dir: Path = None):
    """Download and extract the Daphnet FoG dataset."""
    dest_dir = dest_dir or cfg.raw_data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "dataset_fog_release.zip"

    # Download if not already present
    if not zip_path.exists():
        print(f"Downloading Daphnet dataset to {zip_path} ...")
        urllib.request.urlretrieve(DAPHNET_URL, zip_path)
        print("Download complete.")
    else:
        print(f"Zip file already exists at {zip_path}, skipping download.")

    # Extract
    extract_dir = dest_dir / "dataset_fog_release"
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print("Extraction complete.")
    else:
        print(f"Data already extracted at {extract_dir}, skipping.")

    # List available .txt files (the actual sensor recordings)
    txt_files = sorted(extract_dir.rglob("*.txt"))
    print(f"\nFound {len(txt_files)} recording files:")
    for f in txt_files:
        print(f"  {f.name}")

    return extract_dir


if __name__ == "__main__":
    download_daphnet()

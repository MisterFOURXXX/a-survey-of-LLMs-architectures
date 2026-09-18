#!/usr/bin/env python3
"""
Download the StackSample dataset from Kaggle.

Prerequisites:
    1. Create a Kaggle account at https://www.kaggle.com
    2. Go to Account > API > Create New Token
    3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
       (or set KAGGLE_USERNAME and KAGGLE_KEY environment variables)
    4. Run: pip install kaggle

Usage:
    python scripts/download_data.py
"""
import os
import subprocess
import zipfile
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/datasets/stackoverflow/stacksample"
DATA_DIR = Path("data")
ZIP_NAME = "stacksample.zip"


def setup_kaggle_credentials():
    """Ensure kaggle.json is in the right place."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        os.chmod(kaggle_json, 0o600)
        print(f"✓ Kaggle credentials found at {kaggle_json}")
        return True

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("✓ Using KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        return True

    print("✗ No Kaggle credentials found.")
    print(f"  Download your API token from {DATASET_URL}")
    print(f"  and place kaggle.json in {kaggle_dir}/")
    return False


def download_dataset():
    """Download the StackSample dataset via the Kaggle API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading StackSample dataset...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "stackoverflow/stacksample",
            "-p", str(DATA_DIR),
        ],
        check=True,
    )

    zip_path = DATA_DIR / ZIP_NAME
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"✓ Dataset extracted to {DATA_DIR}/")
    else:
        print(f"✗ Expected zip file not found at {zip_path}")


def verify_files():
    """Check that required CSV files exist."""
    required = ["Questions.csv", "Answers.csv", "Tags.csv"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]

    if missing:
        print(f"✗ Missing files: {missing}")
        return False

    print("✓ All required files present:")
    for f in required:
        size_mb = (DATA_DIR / f).stat().st_size / (1024 * 1024)
        print(f"    {f}: {size_mb:.1f} MB")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("StackSample Dataset Downloader")
    print("=" * 50)

    if not setup_kaggle_credentials():
        raise SystemExit(1)

    download_dataset()

    if verify_files():
        print("\n✓ Ready to use. Run notebooks/01_fine_tuning_comparison.ipynb")
    else:
        print("\n✗ Setup incomplete. Check the errors above.")
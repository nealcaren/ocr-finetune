#!/usr/bin/env python3
"""
Set up tesstrain directory structure for Tesseract fine-tuning.

Creates symlinks from our training data into the tesstrain ground-truth
directory format, and downloads the float eng.traineddata if needed.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TESSTRAIN_DIR = PROJECT_ROOT / "data" / "effocr" / "tesstrain"
TESSDATA_BEST_DIR = PROJECT_ROOT / "data" / "effocr" / "tessdata_best"
TRAINING_DIR = PROJECT_ROOT / "data" / "effocr" / "tesseract_training"
MODEL_NAME = "news_historical"


def clone_tesstrain():
    """Clone tesstrain repo if not present."""
    if TESSTRAIN_DIR.exists() and (TESSTRAIN_DIR / "Makefile").exists():
        print(f"tesstrain already exists at {TESSTRAIN_DIR}")
        return

    print(f"Cloning tesstrain to {TESSTRAIN_DIR}...")
    subprocess.run(
        ["git", "clone", "https://github.com/tesseract-ocr/tesstrain.git", str(TESSTRAIN_DIR)],
        check=True,
    )
    print("tesstrain cloned.")


def download_tessdata_best():
    """Download float eng.traineddata from tessdata_best."""
    TESSDATA_BEST_DIR.mkdir(parents=True, exist_ok=True)
    eng_path = TESSDATA_BEST_DIR / "eng.traineddata"

    if eng_path.exists():
        size_mb = eng_path.stat().st_size / (1024 * 1024)
        print(f"tessdata_best/eng.traineddata already exists ({size_mb:.1f} MB)")
        return

    url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"
    print(f"Downloading float eng.traineddata from {url}...")
    subprocess.run(
        ["curl", "-L", "-o", str(eng_path), url],
        check=True,
    )
    size_mb = eng_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded eng.traineddata ({size_mb:.1f} MB)")


def setup_ground_truth():
    """Symlink training data into tesstrain ground-truth directory."""
    gt_dir = TESSTRAIN_DIR / "data" / f"{MODEL_NAME}-ground-truth"

    # Clean existing
    if gt_dir.exists():
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    train_dir = TRAINING_DIR / "train"
    if not train_dir.exists():
        print(f"ERROR: Training dir not found: {train_dir}")
        print("Run prepare_tesseract_training.py first.")
        return 0

    # Symlink all .png and .gt.txt files from train split
    png_count = 0
    gt_count = 0

    for f in sorted(train_dir.iterdir()):
        if f.suffix == ".png":
            os.symlink(f.resolve(), gt_dir / f.name)
            png_count += 1
        elif f.name.endswith(".gt.txt"):
            os.symlink(f.resolve(), gt_dir / f.name)
            gt_count += 1

    print(f"Symlinked {png_count} PNGs and {gt_count} .gt.txt files to {gt_dir}")
    return png_count


def print_training_command():
    """Print the gmake command to run training."""
    print("\n" + "=" * 70)
    print("tesstrain is ready. Run training with:")
    print("=" * 70)
    print(f"""
cd {TESSTRAIN_DIR}
gmake training \\
    MODEL_NAME={MODEL_NAME} \\
    START_MODEL=eng \\
    TESSDATA={TESSDATA_BEST_DIR} \\
    MAX_ITERATIONS=5000 \\
    TARGET_ERROR_RATE=0.01
""")
    print("Or to run with more iterations:")
    print(f"""
cd {TESSTRAIN_DIR}
gmake training \\
    MODEL_NAME={MODEL_NAME} \\
    START_MODEL=eng \\
    TESSDATA={TESSDATA_BEST_DIR} \\
    MAX_ITERATIONS=10000 \\
    TARGET_ERROR_RATE=0.005
""")
    print("=" * 70)


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"tesstrain dir: {TESSTRAIN_DIR}")
    print(f"tessdata_best: {TESSDATA_BEST_DIR}")
    print(f"Training data: {TRAINING_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print()

    # Step 1: Clone tesstrain
    clone_tesstrain()
    print()

    # Step 2: Download float traineddata
    download_tessdata_best()
    print()

    # Step 3: Symlink ground truth
    count = setup_ground_truth()
    if count == 0:
        return 1

    # Step 4: Print training command
    print_training_command()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Evaluate three Tesseract models on the gold test set and report CER.
Models:
  1. eng           - system baseline
  2. news_historical - silver-trained
  3. news_gold       - gold-trained
"""

import os
import glob
import subprocess
import tempfile
import unicodedata

TEST_DIR = "data/effocr/tesseract_gold_training/test"
TESSDATA_DIR = "data/effocr/tesstrain/data"

MODELS = [
    {
        "name": "eng (baseline)",
        "lang": "eng",
        "tessdata_dir": None,   # use system tessdata
    },
    {
        "name": "news_historical (silver-trained)",
        "lang": "news_historical",
        "tessdata_dir": TESSDATA_DIR,
    },
    {
        "name": "news_gold (gold-trained)",
        "lang": "news_gold",
        "tessdata_dir": TESSDATA_DIR,
    },
]


def edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Roll two rows to save memory
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1,       # deletion
                          curr[j - 1] + 1,   # insertion
                          prev[j - 1] + cost) # substitution
        prev, curr = curr, prev
    return prev[lb]


def normalize(text: str) -> str:
    """Normalize unicode and strip trailing whitespace/newlines."""
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def run_tesseract(png_path: str, lang: str, tessdata_dir: str | None) -> str:
    """Run tesseract on a single PNG and return recognized text."""
    with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
        out_base = f.name  # tesseract appends .txt

    try:
        cmd = ["tesseract", png_path, out_base, "--psm", "7", "-l", lang]
        if tessdata_dir:
            cmd += ["--tessdata-dir", tessdata_dir]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out_txt = out_base + ".txt"
        if os.path.exists(out_txt):
            with open(out_txt) as fh:
                text = fh.read()
            os.unlink(out_txt)
        else:
            text = ""
    except subprocess.TimeoutExpired:
        text = ""
    finally:
        # Clean up temp file if tesseract didn't create .txt
        if os.path.exists(out_base):
            os.unlink(out_base)

    return normalize(text)


def evaluate_model(png_files, gt_map, lang, tessdata_dir):
    total_chars = 0
    total_errors = 0
    failures = 0

    for png in png_files:
        stem = os.path.splitext(png)[0]
        gt = gt_map.get(stem, "")
        if not gt:
            continue

        pred = run_tesseract(png, lang, tessdata_dir)
        errors = edit_distance(gt, pred)
        total_chars += len(gt)
        total_errors += errors

        if pred == "":
            failures += 1

    cer = total_errors / total_chars if total_chars > 0 else float("nan")
    return cer, total_chars, total_errors, failures


def main():
    # Collect all PNG paths and build ground-truth map
    png_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    gt_map = {}
    for png in png_files:
        stem = os.path.splitext(png)[0]
        gt_path = stem + ".gt.txt"
        if os.path.exists(gt_path):
            with open(gt_path) as fh:
                gt_map[stem] = normalize(fh.read())

    n_images = len(png_files)
    n_with_gt = len(gt_map)
    print(f"Test set: {n_images} PNGs, {n_with_gt} with ground truth\n")
    print(f"{'Model':<40}  {'CER':>8}  {'Errors':>8}  {'Chars':>8}  {'Empty':>6}")
    print("-" * 76)

    for model in MODELS:
        cer, chars, errors, failures = evaluate_model(
            png_files, gt_map, model["lang"], model["tessdata_dir"]
        )
        print(
            f"{model['name']:<40}  {cer:>8.4f}  {errors:>8d}  {chars:>8d}  {failures:>6d}"
        )

    print()
    print("CER = character error rate (lower is better)")
    print("Empty = number of images where tesseract returned no text")


if __name__ == "__main__":
    main()

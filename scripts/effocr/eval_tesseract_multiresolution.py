"""Evaluate Tesseract models at multiple resolutions on held-out test pages.

For each test line, downscales the crop to simulate reduced-resolution pages,
then runs each Tesseract model and computes CER against gold-standard text.

This answers: does multi-resolution training actually help at 2800px?

Usage:
    python scripts/effocr/eval_tesseract_multiresolution.py
"""

import json
import subprocess
import tempfile
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"
TEST_DIR = DATA_DIR / "tesseract_100page_v2" / "test"

# Models to compare
MODELS = [
    ("eng (baseline)", {"args": ["-l", "eng"]}),
    ("news_gold_v2 (8pg)", {
        "args": ["--tessdata-dir", str(DATA_DIR / "tesstrain" / "data"), "-l", "news_gold_v2"],
    }),
    ("news_100page (100pg)", {
        "args": ["--tessdata-dir", str(DATA_DIR / "tesstrain" / "data"), "-l", "news_100page"],
    }),
]

# Resolutions to test (scale factors applied to original line crops)
RESOLUTIONS = [
    ("100%", 1.0),
    ("50%", 0.5),
    ("35%", 0.35),
    ("25%", 0.25),
    ("15%", 0.15),
]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1,
                        prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1))
    return dp[n] / m


def get_test_pairs() -> list[tuple[Path, str]]:
    """Get (png_path, ground_truth_text) pairs from test split.

    Only uses original-resolution crops (no *_50pct, *_35pct etc.)
    """
    pairs = []
    for gt_file in sorted(TEST_DIR.glob("*.gt.txt")):
        # Skip multi-res copies and noise
        name = gt_file.name
        if any(f"_{s}pct" in name for s in ["50", "35", "25", "15"]):
            continue
        if "noise" in name:
            continue

        text = gt_file.read_text().strip()
        if len(text) < 3:
            continue

        png_name = name.replace(".gt.txt", ".png")
        png = TEST_DIR / png_name
        if not png.exists():
            continue

        pairs.append((png, text))
    return pairs


def run_tesseract(img_path: str, extra_args: list[str]) -> str:
    cmd = ["tesseract", img_path, "stdout", "--psm", "7"] + extra_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception:
        return ""


def main():
    pairs = get_test_pairs()
    print(f"Test pairs: {len(pairs)} original-resolution lines")
    print(f"Models: {len(MODELS)}")
    print(f"Resolutions: {[r[0] for r in RESOLUTIONS]}")
    print()

    # Check which models exist
    available_models = []
    for model_name, cfg in MODELS:
        if "--tessdata-dir" in cfg["args"]:
            idx = cfg["args"].index("--tessdata-dir")
            tessdata = cfg["args"][idx + 1]
            lang_idx = cfg["args"].index("-l")
            lang = cfg["args"][lang_idx + 1]
            model_path = Path(tessdata) / f"{lang}.traineddata"
            if not model_path.exists():
                print(f"  SKIP {model_name}: {model_path} not found")
                continue
        available_models.append((model_name, cfg))

    if not available_models:
        print("No models available!")
        return

    # Results: {(model, resolution): [cer_values]}
    results = defaultdict(list)

    for i, (png_path, ref_text) in enumerate(pairs):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(pairs)}...")

        img = Image.open(png_path)
        orig_w, orig_h = img.size

        for res_name, scale in RESOLUTIONS:
            # Downscale
            if scale < 1.0:
                new_w = max(1, int(orig_w * scale))
                new_h = max(1, int(orig_h * scale))
                scaled = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                scaled = img

            # Save temp image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                scaled.save(f.name)
                temp_path = f.name

            # Run each model
            for model_name, cfg in available_models:
                hyp = run_tesseract(temp_path, cfg["args"])
                c = cer(ref_text, hyp)
                results[(model_name, res_name)].append(c)

            os.unlink(temp_path)

        img.close()

    # Print results table
    print()
    print("=" * 80)
    print("MULTI-RESOLUTION CER COMPARISON")
    print("=" * 80)
    print()

    # Header
    header = f"{'Resolution':<12}"
    for model_name, _ in available_models:
        header += f" {model_name:>20}"
    print(header)
    print("-" * len(header))

    # Data rows
    for res_name, _ in RESOLUTIONS:
        row = f"{res_name:<12}"
        for model_name, _ in available_models:
            cers = results[(model_name, res_name)]
            avg = sum(cers) / len(cers) if cers else 1.0
            row += f" {avg:>19.2%}"
        print(row)

    print()

    # Also show: how many lines were "perfect" (CER=0) at each resolution
    print("Perfect lines (CER = 0):")
    header = f"{'Resolution':<12}"
    for model_name, _ in available_models:
        header += f" {model_name:>20}"
    print(header)
    print("-" * len(header))

    for res_name, _ in RESOLUTIONS:
        row = f"{res_name:<12}"
        for model_name, _ in available_models:
            cers = results[(model_name, res_name)]
            perfect = sum(1 for c in cers if c == 0)
            total = len(cers)
            row += f" {perfect}/{total} ({perfect/total*100:.0f}%)" if total else " N/A"
            # Pad to match header width
        print(row)

    # Save results
    output = {}
    for (model_name, res_name), cers in results.items():
        key = f"{model_name}@{res_name}"
        output[key] = {
            "avg_cer": sum(cers) / len(cers) if cers else 1.0,
            "median_cer": sorted(cers)[len(cers) // 2] if cers else 1.0,
            "perfect": sum(1 for c in cers if c == 0),
            "total": len(cers),
        }

    output_path = DATA_DIR / "eval_multiresolution_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

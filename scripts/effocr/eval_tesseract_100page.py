"""Evaluate Tesseract models on the 100-page test split.

Compares eng (baseline), news_gold_v2 (8-page fine-tune), and
news_100page (100-page fine-tune with blank examples) on the test set.

Uses --psm 7 (single text line) and computes CER per example.
Also reports blank-detection accuracy for noise/illegible crops.

Usage:
    python scripts/effocr/eval_tesseract_100page.py
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"
TEST_DIR = DATA_DIR / "tesseract_100page_v2" / "test"
TESSDATA_DIR = DATA_DIR / "tesstrain" / "data"
TESSDATA_BEST = DATA_DIR / "tessdata_best"

# Models to evaluate: name -> tessdata path
MODELS = {
    "eng": TESSDATA_BEST,
    "news_gold_v2": TESSDATA_DIR,
    "news_100page": TESSDATA_DIR,
}


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate using edit distance."""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0 if hyp else 0.0

    # Simple Levenshtein
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m] / max(n, 1)


def run_tesseract(image_path: Path, model_name: str, tessdata_dir: Path) -> str:
    """Run tesseract --psm 7 and return stripped output."""
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "7",
             "--tessdata-dir", str(tessdata_dir), "-l", model_name],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: {TEST_DIR} not found. Run build_tesseract_100page.py first.")
        return

    # Collect test pairs (full-res only for speed; skip downscaled _NNpct variants)
    gt_files = sorted(TEST_DIR.glob("*.gt.txt"))
    print(f"Found {len(gt_files)} test gt.txt files total", flush=True)

    pairs = []
    for gt_file in gt_files:
        # Skip downscaled variants — only eval full-res
        stem = gt_file.name.replace(".gt.txt", "")
        if any(stem.endswith(f"_{s}pct") for s in ["15", "25", "35", "50"]):
            continue
        png_name = gt_file.name.replace(".gt.txt", ".png")
        png_file = TEST_DIR / png_name
        if not png_file.exists():
            continue
        ref_text = gt_file.read_text(encoding="utf-8").strip()
        is_blank = (ref_text == "")
        pairs.append({
            "name": gt_file.stem.replace(".gt", ""),
            "png": png_file,
            "ref": ref_text,
            "is_blank": is_blank,
        })

    n_positive = sum(1 for p in pairs if not p["is_blank"])
    n_blank = sum(1 for p in pairs if p["is_blank"])
    print(f"  {n_positive} positive, {n_blank} blank (full-res only)", flush=True)

    # Check which models are available
    available_models = {}
    for name, tessdata in MODELS.items():
        td_file = tessdata / f"{name}.traineddata"
        if td_file.exists():
            available_models[name] = tessdata
            print(f"  Model '{name}': {td_file}", flush=True)
        else:
            print(f"  Model '{name}': NOT FOUND at {td_file}", flush=True)

    if not available_models:
        print("ERROR: No models available for evaluation.", flush=True)
        return

    # Evaluate each model
    results = {}  # model -> list of (ref, hyp, is_blank, cer)
    for model_name, tessdata in available_models.items():
        print(f"\nEvaluating {model_name}...", flush=True)
        model_results = []
        for i, pair in enumerate(pairs):
            hyp = run_tesseract(pair["png"], model_name, tessdata)
            c = cer(pair["ref"], hyp)
            model_results.append({
                "name": pair["name"],
                "ref": pair["ref"],
                "hyp": hyp,
                "is_blank": pair["is_blank"],
                "cer": c,
            })
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(pairs)}...", flush=True)
        results[model_name] = model_results
        print(f"  Done: {len(model_results)} examples evaluated", flush=True)

    # --- Report ---
    print("\n" + "=" * 80)
    print("RESULTS: Tesseract Model Comparison on 100-page Test Split")
    print("=" * 80)

    # Header
    models = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:>18}" for m in models)
    print(header)
    print("-" * len(header))

    # Positive examples (text lines)
    for model_name in models:
        pos_cers = [r["cer"] for r in results[model_name] if not r["is_blank"]]
        if pos_cers:
            avg_cer = sum(pos_cers) / len(pos_cers)
        else:
            avg_cer = float("nan")
        results[model_name].append({"_avg_cer_positive": avg_cer})

    row = f"{'Avg CER (positive)':<30}"
    for model_name in models:
        pos_cers = [r["cer"] for r in results[model_name] if not r.get("is_blank") and "cer" in r]
        avg = sum(pos_cers) / len(pos_cers) if pos_cers else float("nan")
        row += f"{avg:>17.4f}"
    print(row)

    # Median CER
    row = f"{'Median CER (positive)':<30}"
    for model_name in models:
        pos_cers = sorted([r["cer"] for r in results[model_name] if not r.get("is_blank") and "cer" in r])
        if pos_cers:
            mid = len(pos_cers) // 2
            median = pos_cers[mid] if len(pos_cers) % 2 else (pos_cers[mid-1] + pos_cers[mid]) / 2
        else:
            median = float("nan")
        row += f"{median:>17.4f}"
    print(row)

    # Perfect (CER = 0)
    row = f"{'Perfect (CER=0) %':<30}"
    for model_name in models:
        pos_results = [r for r in results[model_name] if not r.get("is_blank") and "cer" in r]
        if pos_results:
            perfect = sum(1 for r in pos_results if r["cer"] == 0.0) / len(pos_results) * 100
        else:
            perfect = 0
        row += f"{perfect:>16.1f}%"
    print(row)

    # Blank examples
    if n_blank > 0:
        print()
        row = f"{'Blank: correct empty %':<30}"
        for model_name in models:
            blank_results = [r for r in results[model_name] if r.get("is_blank") and "cer" in r]
            if blank_results:
                correct = sum(1 for r in blank_results if r["hyp"].strip() == "") / len(blank_results) * 100
            else:
                correct = 0
            row += f"{correct:>16.1f}%"
        print(row)

        row = f"{'Blank: avg output len':<30}"
        for model_name in models:
            blank_results = [r for r in results[model_name] if r.get("is_blank") and "cer" in r]
            if blank_results:
                avg_len = sum(len(r["hyp"]) for r in blank_results) / len(blank_results)
            else:
                avg_len = 0
            row += f"{avg_len:>17.1f}"
        print(row)

    print()
    print(f"Total test examples: {len(pairs)} ({n_positive} positive, {n_blank} blank)", flush=True)


if __name__ == "__main__":
    main()

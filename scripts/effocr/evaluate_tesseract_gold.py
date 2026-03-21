#!/usr/bin/env python3
"""
Evaluate three Tesseract models on the gold-standard test split:
  1. Baseline eng (tessdata_best float model)
  2. news_historical (fine-tuned on EffOCR silver labels)
  3. news_gold (fine-tuned on LLM gold labels)

Test data: data/effocr/tesseract_gold_training/test/
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "effocr" / "tesseract_gold_training" / "test"
TESSTRAIN_DIR = PROJECT_ROOT / "data" / "effocr" / "tesstrain"
TESSDATA_BEST = PROJECT_ROOT / "data" / "effocr" / "tessdata_best"


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def cer(predicted: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return levenshtein_distance(predicted, ground_truth) / len(ground_truth)


def run_tesseract(image_path: Path, model: str, tessdata: Path) -> str:
    """Run tesseract on an image, return predicted text."""
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout",
             "--tessdata-dir", str(tessdata),
             "-l", model, "--psm", "7"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def find_test_pairs(test_dir: Path) -> list[tuple[Path, str]]:
    """Find (png, ground_truth_text) pairs in test dir."""
    pairs = []
    for gt_path in sorted(test_dir.glob("*.gt.txt")):
        stem = gt_path.name.replace(".gt.txt", "")
        png_path = test_dir / f"{stem}.png"
        if png_path.exists():
            text = gt_path.read_text(encoding="utf-8").strip()
            if text:
                pairs.append((png_path, text))
    return pairs


def evaluate(pairs: list, model: str, tessdata: Path) -> dict:
    """Evaluate a model on test pairs."""
    total_cer = 0.0
    total_chars = 0
    perfect = 0
    samples = []

    for i, (png, gt) in enumerate(pairs):
        pred = run_tesseract(png, model, tessdata)
        c = cer(pred, gt)
        total_cer += c * len(gt)
        total_chars += len(gt)
        if pred == gt:
            perfect += 1
        samples.append({"file": png.name, "gt": gt, "pred": pred, "cer": round(c * 100, 2)})

    avg_cer = total_cer / total_chars * 100 if total_chars else 0
    return {
        "model": model,
        "tessdata": str(tessdata),
        "n_samples": len(pairs),
        "avg_cer_pct": round(avg_cer, 2),
        "perfect_matches": perfect,
        "perfect_pct": round(perfect / len(pairs) * 100, 1) if pairs else 0,
        "samples": samples,
    }


def print_results(results: dict, label: str = ""):
    """Print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {label or results['model']} results")
    print(f"{'=' * 60}")
    print(f"  Samples:   {results['n_samples']}")
    print(f"  Avg CER:   {results['avg_cer_pct']:.2f}%")
    print(f"  Perfect:   {results['perfect_matches']}/{results['n_samples']} ({results['perfect_pct']}%)")

    # Worst 5
    worst = sorted(results["samples"], key=lambda x: x["cer"], reverse=True)[:5]
    print(f"\n  Worst 5:")
    for s in worst:
        print(f"    CER={s['cer']:6.2f}%  GT: {s['gt'][:60]}")
        print(f"              Pred: {s['pred'][:60]}")


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: Test dir not found: {TEST_DIR}")
        return 1

    pairs = find_test_pairs(TEST_DIR)
    print(f"Found {len(pairs)} test pairs in {TEST_DIR}")
    if not pairs:
        print("No test pairs found")
        return 1

    # Model configs: (label, model_name, tessdata_dir)
    models = [
        ("Baseline eng (tessdata_best)", "eng", TESSDATA_BEST),
        ("Fine-tuned news_historical (EffOCR labels)", "news_historical", TESSTRAIN_DIR / "data"),
        ("Fine-tuned news_gold (LLM labels)", "news_gold", TESSTRAIN_DIR / "data"),
    ]

    all_results = []
    for label, model, tessdata in models:
        # Check model exists
        model_path = tessdata / f"{model}.traineddata"
        if not model_path.exists():
            print(f"\nWARNING: {model_path} not found, skipping {label}")
            continue
        print(f"\nEvaluating {label}...")
        result = evaluate(pairs, model, tessdata)
        print_results(result, label)
        all_results.append((label, result))

    # Summary comparison table
    if len(all_results) >= 2:
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON TABLE (gold test set, {pairs[0][0].parent})")
        print(f"{'=' * 60}")
        print(f"  {'Model':<45} {'CER':>8} {'Perfect':>10}")
        print(f"  {'-'*45} {'-'*8} {'-'*10}")
        for label, r in all_results:
            print(f"  {label:<45} {r['avg_cer_pct']:>7.2f}% {r['perfect_pct']:>8.1f}%")

        # Improvements relative to baseline
        if all_results[0][1]["avg_cer_pct"] > 0:
            baseline_cer = all_results[0][1]["avg_cer_pct"]
            print(f"\n  Improvements vs baseline ({baseline_cer:.2f}% CER):")
            for label, r in all_results[1:]:
                diff = baseline_cer - r["avg_cer_pct"]
                rel = diff / baseline_cer * 100
                print(f"    {label}: {diff:+.2f}% absolute ({rel:+.1f}% relative)")

    # Save results
    output = {}
    for label, r in all_results:
        key = r["model"]
        output[key] = {k: v for k, v in r.items() if k != "samples"}
    out_path = PROJECT_ROOT / "data" / "effocr" / "tesseract_gold_training" / "eval_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Evaluate baseline eng vs fine-tuned news_historical Tesseract models
on the test split of EffOCR line labels.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "effocr" / "tesseract_training" / "test"
TESSTRAIN_DIR = PROJECT_ROOT / "data" / "effocr" / "tesstrain"
TESSDATA_BEST = PROJECT_ROOT / "data" / "effocr" / "tessdata_best"
SYSTEM_TESSDATA = Path("/opt/homebrew/share/tessdata")


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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return ""


def find_test_pairs(test_dir: Path) -> list[tuple[Path, str]]:
    """Find (png, ground_truth_text) pairs in test dir."""
    pairs = []
    for gt_path in sorted(test_dir.glob("*.gt.txt")):
        png_path = gt_path.parent / f"{gt_path.stem.replace('.gt', '')}.png"
        # gt.txt stem is like "page__line.gt" -> we need "page__line.png"
        stem = gt_path.name.replace(".gt.txt", "")
        png_path = test_dir / f"{stem}.png"
        if png_path.exists():
            text = gt_path.read_text(encoding="utf-8").strip()
            if text:
                pairs.append((png_path, text))
    return pairs


def evaluate(pairs: list, model: str, tessdata: Path, limit: int = None) -> dict:
    """Evaluate a model on test pairs."""
    if limit:
        pairs = pairs[:limit]

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

        if (i + 1) % 100 == 0:
            avg = total_cer / total_chars * 100 if total_chars else 0
            print(f"  [{i+1}/{len(pairs)}] running CER: {avg:.2f}%")

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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Tesseract models on test set")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N test samples")
    args = parser.parse_args()

    if not TEST_DIR.exists():
        print(f"ERROR: Test dir not found: {TEST_DIR}")
        return 1

    pairs = find_test_pairs(TEST_DIR)
    print(f"Found {len(pairs)} test pairs in {TEST_DIR}")

    if not pairs:
        print("No test pairs found")
        return 1

    # Evaluate baseline eng (using tessdata_best float model)
    print(f"\nEvaluating baseline eng (tessdata_best)...")
    baseline = evaluate(pairs, "eng", TESSDATA_BEST, limit=args.limit)
    print_results(baseline, "Baseline eng (tessdata_best)")

    # Evaluate fine-tuned model
    finetuned_model = TESSTRAIN_DIR / "data" / "news_historical.traineddata"
    if finetuned_model.exists():
        print(f"\nEvaluating fine-tuned news_historical...")
        finetuned = evaluate(pairs, "news_historical", TESSTRAIN_DIR / "data", limit=args.limit)
        print_results(finetuned, "Fine-tuned news_historical")

        # Compare
        cer_diff = baseline["avg_cer_pct"] - finetuned["avg_cer_pct"]
        print(f"\n{'=' * 60}")
        print(f"  Improvement: {cer_diff:+.2f}% CER")
        print(f"  Baseline:    {baseline['avg_cer_pct']:.2f}% CER")
        print(f"  Fine-tuned:  {finetuned['avg_cer_pct']:.2f}% CER")
        print(f"{'=' * 60}")

        # Save results
        output = {
            "baseline": {k: v for k, v in baseline.items() if k != "samples"},
            "finetuned": {k: v for k, v in finetuned.items() if k != "samples"},
            "cer_improvement_pct": round(cer_diff, 2),
        }
        out_path = PROJECT_ROOT / "data" / "effocr" / "tesseract_training" / "eval_results.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {out_path}")
    else:
        print(f"\nFine-tuned model not found at {finetuned_model}")
        print("Run tesstrain training first.")

        # Save baseline only
        output = {"baseline": {k: v for k, v in baseline.items() if k != "samples"}}
        out_path = PROJECT_ROOT / "data" / "effocr" / "tesseract_training" / "eval_results.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Baseline results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

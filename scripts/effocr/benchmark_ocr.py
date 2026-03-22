"""Comprehensive OCR benchmark across backends and resolutions.

Runs all available OCR backends at multiple resolutions on the held-out
test pages (from 100-page gold-standard split). Saves results to JSON
for comparison.

Test pages are from DIFFERENT newspapers than training pages.

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/benchmark_ocr.py

Results saved to: data/effocr/benchmark_results.json
"""

import json
import subprocess
import tempfile
import time
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"
VERIFIED_PATH = DATA_DIR / "100page_verified.jsonl"
EXTRACTIONS_DIR = DATA_DIR / "100page_extractions"
RESULTS_PATH = DATA_DIR / "benchmark_results.json"

# Resolutions to test (scale factors applied to original line crops)
RESOLUTIONS = [
    ("100%", 1.0),
    ("50%", 0.5),
    ("35%", 0.35),
    ("25%", 0.25),
    ("15%", 0.15),
]

# Tesseract models
TESSERACT_MODELS = [
    ("tesseract-eng", {"lang": "eng", "tessdata": None}),
]

# Check for fine-tuned models
TESSTRAIN_DATA = DATA_DIR / "tesstrain" / "data"
for model_name in ["news_gold_v2", "news_100page"]:
    model_path = TESSTRAIN_DATA / f"{model_name}.traineddata"
    if model_path.exists():
        TESSERACT_MODELS.append((
            f"tesseract-{model_name}",
            {"lang": model_name, "tessdata": str(TESSTRAIN_DATA)},
        ))


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein distance."""
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


def load_test_lines() -> list[dict]:
    """Load test-split lines with gold-standard transcriptions."""
    lines = []
    for entry in map(json.loads, VERIFIED_PATH.read_text().splitlines()):
        if entry.get("split") != "test":
            continue
        if entry.get("flag") not in ("clean", "partial", "degraded"):
            continue
        if entry.get("confidence", 0) < 0.8:
            continue
        if len(entry.get("transcription", "")) < 3:
            continue
        crop_path = Path(entry["crop_path"])
        if not crop_path.exists():
            # Try relative to extractions dir
            alt = EXTRACTIONS_DIR / "test" / entry["page_id"] / "lines" / f"{entry['line_id']}.png"
            if alt.exists():
                crop_path = alt
            else:
                continue
        lines.append({
            "page_id": entry["page_id"],
            "line_id": entry["line_id"],
            "crop_path": str(crop_path),
            "gold_text": entry["transcription"],
        })
    return lines


def run_tesseract(img_path: str, lang: str, tessdata: str | None) -> str:
    """Run Tesseract on an image, return text."""
    cmd = ["tesseract", img_path, "stdout", "--psm", "7", "-l", lang]
    if tessdata:
        cmd.extend(["--tessdata-dir", tessdata])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception:
        return ""


def run_effocr(img_path: str, effocr_instance) -> str:
    """Run EffOCR on an image, return text."""
    try:
        results = effocr_instance.infer(img_path)
        if results and results[0].text:
            return results[0].text.strip()
    except Exception:
        pass
    return ""


def main():
    # Load test lines
    test_lines = load_test_lines()
    print(f"Test lines: {len(test_lines)} (from held-out test pages)")
    print(f"Test pages: {len(set(l['page_id'] for l in test_lines))}")
    print()

    # Initialize EffOCR if available
    effocr = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from efficient_ocr import EffOCR
        effocr = EffOCR(config={
            "Global": {"skip_line_detection": True},
            "Recognizer": {
                "char": {"model_backend": "onnx", "model_dir": str(DATA_DIR / "models"),
                         "hf_repo_id": "dell-research-harvard/effocr_en/char_recognizer"},
                "word": {"model_backend": "onnx", "model_dir": str(DATA_DIR / "models"),
                         "hf_repo_id": "dell-research-harvard/effocr_en/word_recognizer"},
            },
            "Localizer": {"model_backend": "onnx", "model_dir": str(DATA_DIR / "models"),
                          "hf_repo_id": "dell-research-harvard/effocr_en"},
            "Line": {"model_backend": "onnx", "model_dir": str(DATA_DIR / "models"),
                     "hf_repo_id": "dell-research-harvard/effocr_en"},
        })
        print("EffOCR loaded")
    except Exception as e:
        print(f"EffOCR not available: {e}")

    # Build list of backends
    backends = []
    for name, cfg in TESSERACT_MODELS:
        backends.append(("tesseract", name, cfg))
    if effocr:
        backends.append(("effocr", "effocr-baseline", {}))

    print(f"Backends: {[b[1] for b in backends]}")
    print(f"Resolutions: {[r[0] for r in RESOLUTIONS]}")
    print()

    # Results: {(backend_name, resolution): [cer_values]}
    results = defaultdict(list)
    timings = defaultdict(float)
    counts = defaultdict(int)

    total_evals = len(test_lines) * len(RESOLUTIONS) * len(backends)
    done = 0

    for i, line in enumerate(test_lines):
        img = Image.open(line["crop_path"])
        orig_w, orig_h = img.size
        ref = line["gold_text"]

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

            for backend_type, backend_name, cfg in backends:
                t0 = time.time()

                if backend_type == "tesseract":
                    hyp = run_tesseract(temp_path, cfg["lang"], cfg.get("tessdata"))
                elif backend_type == "effocr":
                    hyp = run_effocr(temp_path, effocr)

                elapsed = time.time() - t0
                c = cer(ref, hyp)
                results[(backend_name, res_name)].append(c)
                timings[(backend_name, res_name)] += elapsed
                counts[(backend_name, res_name)] += 1
                done += 1

            os.unlink(temp_path)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_lines)}] ({done}/{total_evals} evals)")

        img.close()

    # Build output
    print()
    print("=" * 90)
    print("BENCHMARK RESULTS")
    print(f"Test set: {len(test_lines)} lines from {len(set(l['page_id'] for l in test_lines))} held-out pages")
    print("=" * 90)

    # Table: CER
    print()
    print("Average CER:")
    header = f"{'Resolution':<12}"
    for _, backend_name, _ in backends:
        header += f" {backend_name:>22}"
    print(header)
    print("-" * len(header))

    output_data = {
        "test_lines": len(test_lines),
        "test_pages": len(set(l["page_id"] for l in test_lines)),
        "backends": [b[1] for b in backends],
        "resolutions": [r[0] for r in RESOLUTIONS],
        "results": {},
    }

    for res_name, _ in RESOLUTIONS:
        row = f"{res_name:<12}"
        for _, backend_name, _ in backends:
            cers = results[(backend_name, res_name)]
            avg = sum(cers) / len(cers) if cers else 1.0
            row += f" {avg:>21.2%}"
            output_data["results"][f"{backend_name}@{res_name}"] = {
                "avg_cer": round(avg, 4),
                "median_cer": round(sorted(cers)[len(cers) // 2], 4) if cers else 1.0,
                "perfect": sum(1 for c in cers if c == 0),
                "total": len(cers),
                "avg_time_ms": round(timings[(backend_name, res_name)] / max(1, counts[(backend_name, res_name)]) * 1000, 1),
            }
        print(row)

    # Table: Perfect matches
    print()
    print("Perfect lines (CER = 0):")
    header = f"{'Resolution':<12}"
    for _, backend_name, _ in backends:
        header += f" {backend_name:>22}"
    print(header)
    print("-" * len(header))

    for res_name, _ in RESOLUTIONS:
        row = f"{res_name:<12}"
        for _, backend_name, _ in backends:
            cers = results[(backend_name, res_name)]
            perfect = sum(1 for c in cers if c == 0)
            total = len(cers)
            pct = f"{perfect}/{total} ({perfect/total*100:.0f}%)" if total else "N/A"
            row += f" {pct:>22}"
        print(row)

    # Table: Avg time per line
    print()
    print("Avg time per line (ms):")
    header = f"{'Resolution':<12}"
    for _, backend_name, _ in backends:
        header += f" {backend_name:>22}"
    print(header)
    print("-" * len(header))

    for res_name, _ in RESOLUTIONS:
        row = f"{res_name:<12}"
        for _, backend_name, _ in backends:
            key = (backend_name, res_name)
            avg_ms = timings[key] / max(1, counts[key]) * 1000
            row += f" {avg_ms:>20.0f}ms"
        print(row)

    # Save
    RESULTS_PATH.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

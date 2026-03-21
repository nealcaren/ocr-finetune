#!/usr/bin/env python3
"""
Generate OCR labels for line crops using EffOCR.

Runs EffOCR recognition on all full-res line crops from the training pages
(first 8 pages, excluding the 2 test pages). Saves results to a JSONL file.

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/generate_labels.py [--limit N]
"""

import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "effocr"))

DATA_DIR = PROJECT_ROOT / "data" / "effocr"
AS_DIR = DATA_DIR / "as_extractions"
OUTPUT_FILE = DATA_DIR / "line_labels.jsonl"

# Test pages (last 2) — skip these for training
TEST_PAGES = {
    "sn84024055_1923-05-05_seq-2",
    "sn86092050_1937-07-30_seq-1",
}


def reconstruct_line_text(line_data: dict) -> str:
    """Reconstruct line text from word_preds and final_puncs."""
    word_preds = line_data.get("word_preds", [])
    final_puncs = line_data.get("final_puncs", [])
    parts = []
    for i, word in enumerate(word_preds):
        w = word if word else ""
        if i < len(final_puncs) and final_puncs[i]:
            w += final_puncs[i]
        parts.append(w)
    return " ".join(parts)


def get_training_pages():
    """Get page directories for training (exclude test pages)."""
    pages = []
    for page_dir in sorted(AS_DIR.iterdir()):
        if page_dir.is_dir() and page_dir.name not in TEST_PAGES:
            pages.append(page_dir)
    return pages


def load_existing_labels():
    """Load already-processed line keys for resume support."""
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                rec = json.loads(line.strip())
                key = f"{rec['page_id']}_{rec['line_id']}"
                existing.add(key)
    return existing


def main():
    parser = argparse.ArgumentParser(description="Generate OCR labels for line crops")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max lines to process (0 = all)")
    args = parser.parse_args()

    pages = get_training_pages()
    print(f"Training pages: {len(pages)}")
    for p in pages:
        print(f"  {p.name}")

    # Collect all line crops
    all_lines = []
    for page_dir in pages:
        lines_dir = page_dir / "lines"
        if not lines_dir.exists():
            print(f"  WARNING: No lines dir for {page_dir.name}")
            continue
        for crop_path in sorted(lines_dir.glob("line_*.png")):
            line_id = crop_path.stem
            all_lines.append({
                "page_id": page_dir.name,
                "line_id": line_id,
                "crop_path": str(crop_path),
            })

    print(f"\nTotal line crops: {len(all_lines)}")

    # Resume support
    existing = load_existing_labels()
    remaining = [
        l for l in all_lines
        if f"{l['page_id']}_{l['line_id']}" not in existing
    ]
    print(f"Already processed: {len(existing)}")
    print(f"Remaining: {len(remaining)}")

    if args.limit > 0:
        remaining = remaining[:args.limit]
        print(f"Limited to: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    # Initialize EffOCR
    print("\nInitializing EffOCR (ONNX backend)...")
    from efficient_ocr import EffOCR

    hf_repo = "dell-research-harvard/effocr_en"
    effocr = EffOCR(config={
        "Global": {"skip_line_detection": True},
        "Recognizer": {
            "char": {
                "model_backend": "onnx",
                "model_dir": str(DATA_DIR / "models"),
                "hf_repo_id": f"{hf_repo}/char_recognizer",
            },
            "word": {
                "model_backend": "onnx",
                "model_dir": str(DATA_DIR / "models"),
                "hf_repo_id": f"{hf_repo}/word_recognizer",
            },
        },
        "Localizer": {
            "model_backend": "onnx",
            "model_dir": str(DATA_DIR / "models"),
            "hf_repo_id": hf_repo,
        },
        "Line": {
            "model_backend": "onnx",
            "model_dir": str(DATA_DIR / "models"),
            "hf_repo_id": hf_repo,
        },
    })
    print("EffOCR initialized.\n")

    # Process lines
    t0 = time.time()
    success_count = 0
    skip_count = 0
    error_count = 0

    with open(OUTPUT_FILE, "a") as out_f:
        for i, line_info in enumerate(remaining):
            crop_path = line_info["crop_path"]
            page_id = line_info["page_id"]
            line_id = line_info["line_id"]

            try:
                results = effocr.infer(crop_path)

                # Extract text from results
                text_parts = []
                for bbox_result in results:
                    for k in sorted(bbox_result.preds.keys()):
                        line_data = bbox_result.preds[k]
                        text_parts.append(reconstruct_line_text(line_data))
                text = " ".join(text_parts).strip()

                if not text:
                    skip_count += 1
                    continue

                record = {
                    "page_id": page_id,
                    "line_id": line_id,
                    "crop_path": crop_path,
                    "text": text,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                success_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"  ERROR on {page_id}/{line_id}: {e}")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(remaining)}] "
                      f"{success_count} labeled, {skip_count} empty, {error_count} errors "
                      f"({rate:.1f} lines/s)")

    elapsed = time.time() - t0
    total_labeled = len(existing) + success_count
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  New labels: {success_count}")
    print(f"  Empty (skipped): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total labeled: {total_labeled}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

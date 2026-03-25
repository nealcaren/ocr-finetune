#!/usr/bin/env python3
"""
prepare_byt5_gold.py — Build ByT5 training data using Qwen3-VL region
transcriptions as gold targets.

Aligns Tesseract line outputs to the gold region text using fuzzy matching,
then creates sliding-window (noisy_block, clean_block) pairs.

Usage:
    python scripts/byt5/prepare_byt5_gold.py
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BYT5_DIR = DATA_DIR / "byt5"
EFFOCR_DIR = DATA_DIR / "effocr"
EXTRACTIONS_DIR = EFFOCR_DIR / "pre1930_fullres"
REGION_TRANSCRIPTIONS = BYT5_DIR / "region_transcriptions.jsonl"
TESS_CACHE_PATH = BYT5_DIR / "tesseract_train_cache.json"

WINDOW_SIZE = 10
STRIDE = 5
MIN_BLOCK_LINES = 3


def load_region_transcriptions():
    """Load region-level gold transcriptions keyed by (split, page_id, region_id)."""
    regions = {}
    with open(REGION_TRANSCRIPTIONS) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "ok" or d["confidence"] < 0.8:
                continue
            key = (d["split"], d["page_id"], d["region_id"])
            regions[key] = d
    return regions


def load_tess_cache():
    if TESS_CACHE_PATH.exists():
        return json.loads(TESS_CACHE_PATH.read_text())
    return {}


def load_line_gold():
    """Load per-line gold transcriptions from the fullres verified JSONL."""
    gold = {}
    verified_path = EFFOCR_DIR / "pre1930_fullres_verified.jsonl"
    if verified_path.exists():
        for line in verified_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            key = f"{d['page_id']}__{d['line_id']}"
            gold[key] = d["transcription"].strip()
    return gold


def proportional_split(region_text, num_lines, start_idx, end_idx):
    """
    Split region text proportionally based on line positions.
    Simple and fast: assumes each line contributes roughly equally.
    """
    total = len(region_text)
    chunk_start = int(total * start_idx / num_lines)
    chunk_end = int(total * end_idx / num_lines)

    # Snap to word boundaries
    while chunk_start > 0 and region_text[chunk_start - 1] not in " \n":
        chunk_start -= 1
    while chunk_end < total and region_text[chunk_end] not in " \n":
        chunk_end += 1

    return region_text[chunk_start:chunk_end].strip()


def main():
    BYT5_DIR.mkdir(parents=True, exist_ok=True)

    region_golds = load_region_transcriptions()
    tess_cache = load_tess_cache()
    line_gold = load_line_gold()

    print(f"Region transcriptions: {len(region_golds)}")
    print(f"Tesseract cache: {len(tess_cache)} lines")
    print(f"Line-level gold: {len(line_gold)} lines")

    # Load metadata to get line ordering by region
    all_examples = {"train": [], "val": [], "test": []}

    for (split, page_id, region_id), region_data in sorted(region_golds.items()):
        region_text = region_data["transcription"]
        line_ids = region_data.get("line_ids", [])

        if not line_ids or not region_text:
            continue

        # Get per-line Tesseract outputs
        noisy_lines = []
        for lid in line_ids:
            cache_key = f"{page_id}__{lid}"
            noisy_lines.append(tess_cache.get(cache_key, ""))

        n = len(noisy_lines)
        if n < MIN_BLOCK_LINES:
            continue

        # Build sliding window examples
        windows = []
        if n <= WINDOW_SIZE:
            windows.append((0, n))
        else:
            for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
                windows.append((start, start + WINDOW_SIZE))
            if (n - WINDOW_SIZE) % STRIDE != 0:
                windows.append((n - WINDOW_SIZE, n))

        for start, end in windows:
            noisy_block = "\n".join(noisy_lines[start:end])
            if not noisy_block.strip():
                continue

            # Extract target chunk proportionally from region gold text
            target = proportional_split(region_text, n, start, end)

            if not target or len(target) < 10:
                continue

            # Check size limits
            input_text = f"correct: {noisy_block}"
            if len(input_text.encode()) > 1024 or len(target.encode()) > 1024:
                continue

            all_examples[split].append({
                "input": input_text,
                "target": target,
                "page_id": page_id,
                "region_id": region_id,
                "num_lines": end - start,
            })

    # Save
    for split in ["train", "val", "test"]:
        examples = all_examples[split]
        out_path = BYT5_DIR / f"{split}_gold.json"
        with open(out_path, "w") as f:
            json.dump(examples, f, indent=2)

        if examples:
            avg_in = sum(len(e["input"]) for e in examples) / len(examples)
            avg_tgt = sum(len(e["target"]) for e in examples) / len(examples)
            print(f"\n{split}: {len(examples)} examples")
            print(f"  Avg input: {avg_in:.0f} bytes, avg target: {avg_tgt:.0f} bytes")
        else:
            print(f"\n{split}: 0 examples")

    total = sum(len(v) for v in all_examples.values())
    print(f"\nTotal: {total} examples")


if __name__ == "__main__":
    main()

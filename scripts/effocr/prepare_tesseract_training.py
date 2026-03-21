#!/usr/bin/env python3
"""
Prepare Tesseract training data from EffOCR line labels.

Reads line_labels.jsonl, creates .gt.txt files alongside copied PNGs,
and splits into train/val/test sets.
"""
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "effocr" / "line_labels.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "effocr" / "tesseract_training"

# Config
MIN_TEXT_LENGTH = 3
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO = 0.10 (remainder)
SEED = 42


def load_labels(labels_path: Path) -> list[dict]:
    """Load, deduplicate, and filter line labels from JSONL."""
    seen_keys = set()
    labels = []
    skipped_empty = 0
    skipped_short = 0
    skipped_missing = 0
    skipped_dup = 0

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            text = record.get("text", "").strip()
            crop_path = Path(record["crop_path"])

            # Deduplicate by page_id + line_id
            key = f"{record['page_id']}__{record['line_id']}"
            if key in seen_keys:
                skipped_dup += 1
                continue
            seen_keys.add(key)

            if not text:
                skipped_empty += 1
                continue
            if len(text) < MIN_TEXT_LENGTH:
                skipped_short += 1
                continue
            if not crop_path.exists():
                skipped_missing += 1
                continue

            labels.append(record)

    print(f"Loaded {len(labels)} valid labels")
    print(f"Skipped: {skipped_dup} duplicates, {skipped_empty} empty, {skipped_short} short (<{MIN_TEXT_LENGTH} chars), {skipped_missing} missing files")
    return labels


def split_labels(labels: list[dict]) -> tuple[list, list, list]:
    """Split labels into train/val/test."""
    random.seed(SEED)
    shuffled = labels.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def write_split(records: list[dict], split_dir: Path, split_name: str):
    """Write PNG + .gt.txt pairs into a split directory."""
    split_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        crop_path = Path(record["crop_path"])
        text = record["text"].strip()
        page_id = record["page_id"]
        line_id = record["line_id"]

        # Use page_id prefix to avoid collisions across pages
        unique_stem = f"{page_id}__{line_id}"

        # Copy PNG
        dest_png = split_dir / f"{unique_stem}.png"
        shutil.copy2(crop_path, dest_png)

        # Write ground truth (single line, no trailing newline)
        gt_path = split_dir / f"{unique_stem}.gt.txt"
        gt_path.write_text(text, encoding="utf-8")

    print(f"  {split_name}: wrote {len(records)} pairs to {split_dir}")


def main():
    print(f"Labels file: {LABELS_PATH}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print()

    if not LABELS_PATH.exists():
        print(f"ERROR: Labels file not found: {LABELS_PATH}")
        return 1

    labels = load_labels(LABELS_PATH)
    if not labels:
        print("ERROR: No valid labels found")
        return 1

    train, val, test = split_labels(labels)

    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    write_split(train, OUTPUT_DIR / "train", "train")
    write_split(val, OUTPUT_DIR / "val", "val")
    write_split(test, OUTPUT_DIR / "test", "test")

    # Write manifest
    manifest = {
        "total": len(labels),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": SEED,
        "min_text_length": MIN_TEXT_LENGTH,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

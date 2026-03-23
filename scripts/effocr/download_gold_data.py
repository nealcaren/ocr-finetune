#!/usr/bin/env python3
"""
download_gold_data.py — Download gold-standard dataset from HuggingFace
and extract images + JSONL for EffOCR training.

Downloads NealCaren/newspaper-ocr-gold (multi-resolution parquet dataset),
saves full-resolution images to disk, and creates verified_lines.jsonl
compatible with build_training_data.py.

Usage:
    python scripts/effocr/download_gold_data.py --output-dir /path/to/gold_data
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save gold data")
    parser.add_argument("--repo", type=str, default="NealCaren/newspaper-ocr-gold",
                        help="HuggingFace dataset repo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo}...")
    ds = load_dataset(args.repo)

    jsonl_path = output_dir / "verified_lines.jsonl"
    total_images = 0

    with open(jsonl_path, "w") as jsonl_f:
        for split_name in ["train", "val", "test"]:
            if split_name not in ds:
                continue

            split = ds[split_name]
            # Filter to full resolution only (we generate multi-res on the fly)
            full_res = split.filter(lambda x: x["resolution"] == "full")
            print(f"\n{split_name}: {len(full_res)} full-res lines (of {len(split)} total)")

            split_dir = output_dir / split_name
            for row in full_res:
                page_id = row["page_id"]
                line_id = row["line_id"]

                # Save image
                page_dir = split_dir / page_id / "lines"
                page_dir.mkdir(parents=True, exist_ok=True)
                img_path = page_dir / f"{line_id}.png"

                if not img_path.exists():
                    row["image"].save(str(img_path))

                # Write JSONL entry
                entry = {
                    "split": split_name,
                    "page_id": page_id,
                    "line_id": line_id,
                    "crop_path": str(img_path),
                    "transcription": row["transcription"],
                    "confidence": row["confidence"],
                    "flag": row["flag"],
                }
                jsonl_f.write(json.dumps(entry) + "\n")
                total_images += 1

    print(f"\nDone!")
    print(f"  Images: {total_images}")
    print(f"  JSONL:  {jsonl_path}")


if __name__ == "__main__":
    main()

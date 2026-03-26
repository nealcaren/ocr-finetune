#!/usr/bin/env python3
"""
build_training_data_fast.py — Build EffOCR training data using localizer only.

Instead of running full EffOCR inference (slow KNN recognition), this script:
1. Runs just the localizer (YOLO) to get char/word bounding boxes (fast)
2. Assigns character labels from gold transcriptions by left-to-right alignment
3. Saves as EffOCR ImageFolder format

~100x faster than build_training_data.py since it skips the recognizer.

Usage:
    python scripts/effocr/build_training_data_fast.py \
        --gold-jsonl /path/to/verified_lines.jsonl \
        --image-dir /path/to/gold_data/ \
        --output-dir /path/to/training_data/ \
        --model-dir /path/to/models/ \
        --scales 1.0,0.5,0.35,0.25
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "effocr"))

DATA_DIR = PROJECT_ROOT / "data" / "effocr"
PROGRESS_FILE = DATA_DIR / "training_data" / "progress_fast.json"


def str_to_ord_str(s: str) -> str:
    """Convert a string to underscore-joined ordinal values (EffOCR convention)."""
    return "_".join(str(ord(c)) for c in s)


def downscale_crop(img, scale):
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def align_chars_to_text(char_crops_with_bboxes, gold_text):
    """
    Align detected character crops to gold transcription text.

    Characters are sorted left-to-right by x-coordinate.
    Gold text characters (excluding spaces) are assigned in order.

    Returns list of (char_crop, label) pairs, or None for unaligned chars.
    """
    if not char_crops_with_bboxes or not gold_text:
        return []

    # Sort chars by x position (left to right)
    sorted_chars = sorted(char_crops_with_bboxes, key=lambda c: c[1][0])

    # Get non-space characters from gold text
    gold_chars = [c for c in gold_text if c != " "]

    results = []
    for i, (crop, bbox) in enumerate(sorted_chars):
        if i < len(gold_chars):
            results.append((crop, gold_chars[i]))
        else:
            results.append((crop, None))

    return results


def align_words_to_text(word_crops_with_bboxes, gold_text):
    """
    Align detected word crops to gold transcription words.

    Words sorted left-to-right, gold words assigned in order.
    """
    if not word_crops_with_bboxes or not gold_text:
        return []

    sorted_words = sorted(word_crops_with_bboxes, key=lambda w: w[1][0])
    gold_words = gold_text.split()

    results = []
    for i, (crop, bbox) in enumerate(sorted_words):
        if i < len(gold_words):
            results.append((crop, gold_words[i]))
        else:
            results.append((crop, None))

    return results


def main():
    parser = argparse.ArgumentParser(description="Build EffOCR training data (fast, localizer-only)")
    parser.add_argument("--gold-jsonl", type=str, required=True,
                        help="Path to verified_lines.jsonl")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Base directory for line crop images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for training data")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory for EffOCR ONNX models")
    parser.add_argument("--scales", type=str, default="1.0,0.5,0.35,0.25",
                        help="Comma-separated downscale factors")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    image_base_dir = Path(args.image_dir)
    char_data_dir = Path(args.output_dir) / "char"
    word_data_dir = Path(args.output_dir) / "word"
    model_dir = Path(args.model_dir)
    progress_file = Path(args.output_dir) / "progress_fast.json"

    # Load labels
    labels = []
    with open(args.gold_jsonl) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("split") != "train":
                continue
            if entry.get("flag", "") not in ("clean", "partial", "degraded"):
                continue
            if entry.get("confidence", 0) < 0.8:
                continue
            if len(entry.get("transcription", "")) < 3:
                continue
            # Resolve image path
            crop = Path(entry["crop_path"])
            if not crop.is_absolute():
                crop = image_base_dir / crop
            if not crop.exists():
                alt = image_base_dir / entry["split"] / entry["page_id"] / "lines" / f"{entry['line_id']}.png"
                if alt.exists():
                    entry["crop_path"] = str(alt)
                else:
                    continue
            else:
                entry["crop_path"] = str(crop)
            labels.append(entry)

    print(f"Loaded {len(labels)} labeled lines")

    # Resume
    if progress_file.exists():
        processed = set(json.loads(progress_file.read_text()).get("processed", []))
    else:
        processed = set()

    remaining = [l for l in labels
                 if f"{l.get('page_id', '')}_{l.get('line_id', '')}" not in processed]
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Scales: {scales}")

    if args.limit > 0:
        remaining = remaining[:args.limit]

    if not remaining:
        print("Nothing to do!")
        return

    char_data_dir.mkdir(parents=True, exist_ok=True)
    word_data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize localizer ONLY (no recognizer)
    print("\nInitializing EffOCR localizer (ONNX)...")
    from efficient_ocr.detection import LocalizerModel
    from efficient_ocr.utils import DEFAULT_CONFIG
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["Localizer"]["model_backend"] = "onnx"
    config["Localizer"]["model_dir"] = str(model_dir)
    config["Localizer"]["hf_repo_id"] = "dell-research-harvard/effocr_en"
    localizer = LocalizerModel(config)
    print("Localizer ready.\n")

    t0 = time.time()
    total_chars = 0
    total_words = 0
    error_count = 0

    for i, label_rec in enumerate(remaining):
        crop_path = label_rec["crop_path"]
        page_id = label_rec["page_id"]
        line_id = label_rec["line_id"]
        gold_text = label_rec["transcription"].strip()
        line_key = f"{page_id}_{line_id}"

        try:
            line_img = Image.open(crop_path).convert("RGB")

            # Run localizer only — get char and word bboxes
            line_results = [(line_img, (0, 0, line_img.size[0], line_img.size[1]))]
            localization = localizer.run_simple(line_results, thresh=0.75)

            if not localization or not localization[0]:
                processed.add(line_key)
                continue

            # localization[0] = list of (word_result, char_results) tuples
            # Collect all char crops
            all_chars = []
            all_words = []
            for item in localization[0]:
                if isinstance(item, tuple) and len(item) == 2:
                    word_result, char_results = item
                    if word_result:
                        all_words.append(word_result)  # (word_img, (x0,y0,x1,y1))
                    if char_results:
                        all_chars.extend(char_results)  # [(char_img, (x0,y0,x1,y1)), ...]

            # Align chars to gold text
            char_labels = align_chars_to_text(all_chars, gold_text)
            word_labels = align_words_to_text(all_words, gold_text)

            # Save character crops
            for crop, label in char_labels:
                if label is None or not label.strip():
                    continue
                char_ord = str(ord(label))
                char_dir = char_data_dir / char_ord
                char_dir.mkdir(parents=True, exist_ok=True)

                pil_img = crop if isinstance(crop, Image.Image) else Image.fromarray(np.array(crop))
                for scale in scales:
                    scaled = downscale_crop(pil_img, scale)
                    if scaled.width < 4 or scaled.height < 4:
                        continue
                    scale_tag = f"_{int(scale * 100)}pct" if len(scales) > 1 else ""
                    fname = f"PAIRED_{page_id}_{line_id}_c{total_chars:06d}{scale_tag}.png"
                    scaled.save(str(char_dir / fname))
                    total_chars += 1

            # Save word crops
            for crop, label in word_labels:
                if label is None or len(label) < 1:
                    continue
                word_ord = str_to_ord_str(label)
                word_dir = word_data_dir / word_ord
                word_dir.mkdir(parents=True, exist_ok=True)

                pil_img = crop if isinstance(crop, Image.Image) else Image.fromarray(np.array(crop))
                for scale in scales:
                    scaled = downscale_crop(pil_img, scale)
                    if scaled.width < 8 or scaled.height < 4:
                        continue
                    scale_tag = f"_{int(scale * 100)}pct" if len(scales) > 1 else ""
                    fname = f"PAIRED_{page_id}_{line_id}_w{total_words:06d}{scale_tag}.png"
                    scaled.save(str(word_dir / fname))
                    total_words += 1

            processed.add(line_key)

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"  ERROR on {line_key}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(remaining)}] "
                  f"{total_chars} chars, {total_words} words, {error_count} errors "
                  f"({rate:.1f} lines/s)")
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            progress_file.write_text(json.dumps({"processed": sorted(processed)}))

    # Final save
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps({"processed": sorted(processed)}))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(remaining) / elapsed:.1f} lines/s)")
    print(f"  Character crops: {total_chars}")
    print(f"  Word crops: {total_words}")
    print(f"  Errors: {error_count}")

    char_classes = len([d for d in char_data_dir.iterdir() if d.is_dir()])
    word_classes = len([d for d in word_data_dir.iterdir() if d.is_dir()])
    print(f"  Unique char classes: {char_classes}")
    print(f"  Unique word classes: {word_classes}")


if __name__ == "__main__":
    main()

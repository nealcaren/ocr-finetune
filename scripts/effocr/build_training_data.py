#!/usr/bin/env python3
"""
Build character and word training data for EfficientOCR fine-tuning.

Process:
1. Read labeled lines from line_labels.jsonl
2. Run EffOCR on each line crop to get char/word crops + predictions
3. Downscale crops to target resolution (35%) to simulate reduced quality
4. Save into EffOCR's ImageFolder structure: <ascii_code>/PAIRED_<name>.png

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/build_training_data.py [--limit N] [--scale 0.35]
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
LABELS_FILE = DATA_DIR / "line_labels.jsonl"
CHAR_DATA_DIR = DATA_DIR / "training_data" / "char"
WORD_DATA_DIR = DATA_DIR / "training_data" / "word"
PROGRESS_FILE = DATA_DIR / "training_data" / "progress.json"


def str_to_ord_str(s: str) -> str:
    """Convert a string to underscore-joined ordinal values (EffOCR convention)."""
    return "_".join(str(ord(c)) for c in s)


def ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array crop to PIL Image."""
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def downscale_crop(img: Image.Image, scale: float) -> Image.Image:
    """Downscale a crop image to simulate lower resolution."""
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


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


def assign_char_labels(line_data: dict) -> list:
    """
    Assign character labels from word_preds using overlaps mapping.
    Returns list of single-char labels (one per char crop), or None for unassignable.
    """
    word_preds = line_data.get("word_preds", [])
    overlaps = line_data.get("overlaps", [])
    chars = line_data.get("chars", [])

    labels = [None] * len(chars)
    for word_idx, char_indices in enumerate(overlaps):
        if word_idx >= len(word_preds) or not word_preds[word_idx]:
            continue
        word_text = word_preds[word_idx]
        for pos, char_idx in enumerate(char_indices):
            if char_idx < len(labels) and pos < len(word_text):
                labels[char_idx] = word_text[pos]
    return labels


def load_progress():
    """Load set of already-processed line keys."""
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("processed", []))
    return set()


def save_progress(processed: set):
    """Save progress."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({"processed": sorted(processed)}))


def main():
    parser = argparse.ArgumentParser(description="Build EffOCR training data")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max lines to process (0 = all)")
    parser.add_argument("--scale", type=float, default=0.35,
                        help="Downscale factor for crops (default: 0.35)")
    args = parser.parse_args()

    # Load labels
    if not LABELS_FILE.exists():
        print(f"ERROR: Labels file not found: {LABELS_FILE}")
        print("Run generate_labels.py first.")
        sys.exit(1)

    labels = []
    with open(LABELS_FILE) as f:
        for line in f:
            labels.append(json.loads(line.strip()))
    print(f"Loaded {len(labels)} labeled lines")

    # Resume support
    processed = load_progress()
    remaining = [
        l for l in labels
        if f"{l['page_id']}_{l['line_id']}" not in processed
    ]
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(remaining)}")

    if args.limit > 0:
        remaining = remaining[:args.limit]
        print(f"Limited to: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    # Create output dirs
    CHAR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORD_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    total_chars = 0
    total_words = 0
    error_count = 0

    for i, label_rec in enumerate(remaining):
        crop_path = label_rec["crop_path"]
        page_id = label_rec["page_id"]
        line_id = label_rec["line_id"]
        line_key = f"{page_id}_{line_id}"

        try:
            # Run EffOCR to get char/word crops
            results = effocr.infer(crop_path)

            for bbox_result in results:
                for k in sorted(bbox_result.preds.keys()):
                    line_data = bbox_result.preds[k]

                    # --- Character crops ---
                    char_labels = assign_char_labels(line_data)
                    chars = line_data.get("chars", [])
                    for c_idx, (char_img, char_bbox) in enumerate(chars):
                        label = char_labels[c_idx] if c_idx < len(char_labels) else None
                        if label is None:
                            continue
                        # Use ASCII ordinal as folder name
                        char_ord = str(ord(label))
                        char_dir = CHAR_DATA_DIR / char_ord
                        char_dir.mkdir(parents=True, exist_ok=True)

                        # Convert to PIL and downscale
                        pil_img = ndarray_to_pil(char_img)
                        pil_img = downscale_crop(pil_img, args.scale)

                        # Save with PAIRED prefix (EffOCR convention for real data)
                        fname = f"PAIRED_{page_id}_{line_id}_c{c_idx:03d}.png"
                        pil_img.save(str(char_dir / fname))
                        total_chars += 1

                    # --- Word crops ---
                    word_preds = line_data.get("word_preds", [])
                    words = line_data.get("words", [])
                    final_puncs = line_data.get("final_puncs", [])
                    for w_idx, (word_img, word_bbox) in enumerate(words):
                        if w_idx >= len(word_preds) or not word_preds[w_idx]:
                            continue
                        word_text = word_preds[w_idx]
                        # Add punctuation if present
                        if w_idx < len(final_puncs) and final_puncs[w_idx]:
                            word_text += final_puncs[w_idx]

                        # Use ordinal string as folder name
                        word_ord = str_to_ord_str(word_text)
                        word_dir = WORD_DATA_DIR / word_ord
                        word_dir.mkdir(parents=True, exist_ok=True)

                        # Convert to PIL and downscale
                        pil_img = ndarray_to_pil(word_img)
                        pil_img = downscale_crop(pil_img, args.scale)

                        fname = f"PAIRED_{page_id}_{line_id}_w{w_idx:03d}.png"
                        pil_img.save(str(word_dir / fname))
                        total_words += 1

            processed.add(line_key)

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"  ERROR on {line_key}: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(remaining)}] "
                  f"{total_chars} chars, {total_words} words, {error_count} errors "
                  f"({rate:.1f} lines/s)")
            save_progress(processed)

    # Final save
    save_progress(processed)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Character crops: {total_chars}")
    print(f"  Word crops: {total_words}")
    print(f"  Errors: {error_count}")
    print(f"  Char data dir: {CHAR_DATA_DIR}")
    print(f"  Word data dir: {WORD_DATA_DIR}")

    # Count unique classes
    char_classes = len([d for d in CHAR_DATA_DIR.iterdir() if d.is_dir()])
    word_classes = len([d for d in WORD_DATA_DIR.iterdir() if d.is_dir()])
    print(f"  Unique char classes: {char_classes}")
    print(f"  Unique word classes: {word_classes}")


if __name__ == "__main__":
    main()

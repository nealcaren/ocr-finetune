#!/usr/bin/env python3
"""
Run EfficientOCR on full-resolution JP2 newspaper scans.

Extracts line crops, word crops, char crops, and OCR text for each page,
saving all intermediate data for downstream processing.

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/run_effocr_extraction.py [--pages N] [--scale SCALE]
"""

import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "effocr"))

def _ndarray_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a numpy array to uint8 for PIL saving.

    Handles float32/float64 arrays (assumed 0-1 or 0-255 range)
    and squeezes leading singleton dims like (1, H, W, C) -> (H, W, C).
    """
    # Squeeze leading singleton dimensions
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


DATA_DIR = PROJECT_ROOT / "data" / "effocr"
PILOT_PAGES_JSON = DATA_DIR / "pilot_pages.json"
OUTPUT_DIR = DATA_DIR / "extractions"


def make_page_id(entry: dict) -> str:
    """Create a unique page ID from pilot_pages.json entry."""
    lccn = entry["lccn"]
    date = entry["date"]
    seq = Path(entry["path"]).stem  # e.g. "seq-4"
    return f"{lccn}_{date}_{seq}"


def convert_jp2_to_jpg(jp2_path: str, scale: float = 1.0) -> str:
    """Convert JP2 to a temporary JPG file. Optionally downscale."""
    img = Image.open(jp2_path)
    if scale < 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Downscaled from {Image.open(jp2_path).size} to {img.size}")
    # Save to a temp file that persists until we're done
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=95)
    print(f"  Converted JP2 -> JPG: {img.size[0]}x{img.size[1]}, {Path(tmp.name).stat().st_size / 1e6:.1f} MB")
    return tmp.name


def reconstruct_line_text(line_data: dict) -> str:
    """
    Reconstruct line text from word_preds and final_puncs.
    Each word gets its final_punc appended, then joined by spaces.
    """
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

    overlaps[i] = list of char indices belonging to word i.
    word_preds[i] = the recognized text for word i.
    final_puncs[i] = punctuation after word i (already stripped from chars).

    Returns a list of (char_index, label) tuples.
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


def process_page(effocr, entry: dict, scale: float = 1.0):
    """Process a single page: run EffOCR and save all outputs."""
    page_id = make_page_id(entry)
    jp2_path = entry["path"]
    page_dir = OUTPUT_DIR / page_id

    # Resume check
    metadata_path = page_dir / "metadata.json"
    if metadata_path.exists():
        print(f"  Already processed, skipping: {page_id}")
        return

    print(f"\nProcessing: {page_id}")
    print(f"  JP2: {jp2_path}")

    # Convert JP2 to JPG
    jpg_path = convert_jp2_to_jpg(jp2_path, scale=scale)

    # Load full image for line cropping
    full_img = Image.open(jpg_path)

    # Run EffOCR
    t0 = time.time()
    try:
        results = effocr.infer(jpg_path)
    except Exception as e:
        print(f"  ERROR during inference: {e}")
        Path(jpg_path).unlink(missing_ok=True)
        raise
    elapsed = time.time() - t0
    print(f"  Inference took {elapsed:.1f}s, got {len(results)} bbox regions")

    # Create output directories
    lines_dir = page_dir / "lines"
    words_dir = page_dir / "words"
    chars_dir = page_dir / "chars"
    for d in [lines_dir, words_dir, chars_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Process results
    all_lines = []
    global_line_idx = 0
    full_text_parts = []

    for bbox_idx, result in enumerate(results):
        preds = result.preds  # dict keyed by line_idx (int)

        for line_idx in sorted(preds.keys()):
            line_data = preds[line_idx]
            line_id = f"line_{global_line_idx:04d}"

            # Debug: print keys on first line to verify preds structure
            if global_line_idx == 0:
                print(f"  Preds keys for first line: {list(line_data.keys())}")

            # Reconstruct text
            line_text = reconstruct_line_text(line_data)
            full_text_parts.append(line_text)

            # Save line crop from full image using bbox
            line_bbox = line_data.get("bbox")
            line_crop_path = None
            if line_bbox is not None:
                y0, x0, y1, x1 = line_bbox  # (y0, x0, y1, x1)
                # PIL crop takes (left, upper, right, lower)
                try:
                    line_crop = full_img.crop((x0, y0, x1, y1))
                    lcp = lines_dir / f"{line_id}.png"
                    line_crop.save(str(lcp))
                    line_crop_path = str(lcp.relative_to(page_dir))
                except Exception as e:
                    print(f"  Warning: failed to crop line {line_id}: {e}")

            # Save word crops
            word_entries = []
            word_imgs = line_data.get("words", [])
            word_preds = line_data.get("word_preds", [])
            for w_idx, (word_img, word_bbox) in enumerate(word_imgs):
                word_id = f"{line_id}_w{w_idx:03d}"
                wp = words_dir / f"{word_id}.png"
                try:
                    if isinstance(word_img, np.ndarray):
                        img_to_save = _ndarray_to_uint8(word_img)
                        Image.fromarray(img_to_save).save(str(wp))
                    word_text = word_preds[w_idx] if w_idx < len(word_preds) else ""
                    word_entries.append({
                        "word_id": word_id,
                        "text": word_text,
                        "bbox": list(word_bbox) if word_bbox is not None else None,
                        "path": str(wp.relative_to(page_dir)),
                    })
                except Exception as e:
                    print(f"  Warning: failed to save word {word_id}: {e}")

            # Save char crops with labels
            char_entries = []
            char_imgs = line_data.get("chars", [])
            char_labels = assign_char_labels(line_data)
            for c_idx, (char_img, char_bbox) in enumerate(char_imgs):
                char_id = f"{line_id}_c{c_idx:03d}"
                cp = chars_dir / f"{char_id}.png"
                try:
                    if isinstance(char_img, np.ndarray):
                        img_to_save = _ndarray_to_uint8(char_img)
                        Image.fromarray(img_to_save).save(str(cp))
                    label = char_labels[c_idx] if c_idx < len(char_labels) else None
                    char_entries.append({
                        "char_id": char_id,
                        "label": label,
                        "bbox": list(char_bbox) if char_bbox is not None else None,
                        "path": str(cp.relative_to(page_dir)),
                    })
                except Exception as e:
                    print(f"  Warning: failed to save char {char_id}: {e}")

            all_lines.append({
                "line_id": line_id,
                "line_index": global_line_idx,
                "bbox_idx": bbox_idx,
                "text": line_text,
                "bbox": list(line_bbox) if line_bbox is not None else None,
                "para_end": line_data.get("para_end", False),
                "line_crop_path": line_crop_path,
                "num_words": len(word_entries),
                "num_chars": len(char_entries),
                "words": word_entries,
                "chars": char_entries,
            })
            global_line_idx += 1

    # Build metadata
    full_text = "\n".join(full_text_parts)
    metadata = {
        "page_id": page_id,
        "jp2_path": jp2_path,
        "scale": scale,
        "image_size": [full_img.width, full_img.height],
        "num_bbox_regions": len(results),
        "num_lines": len(all_lines),
        "num_words": sum(l["num_words"] for l in all_lines),
        "num_chars": sum(l["num_chars"] for l in all_lines),
        "inference_time_s": round(elapsed, 1),
        "full_text": full_text,
        "lines": all_lines,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Cleanup temp JPG
    Path(jpg_path).unlink(missing_ok=True)

    print(f"  Done: {len(all_lines)} lines, "
          f"{metadata['num_words']} words, "
          f"{metadata['num_chars']} chars")
    print(f"  Output: {page_dir}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Run EffOCR extraction on pilot pages")
    parser.add_argument("--pages", type=int, default=1,
                        help="Number of pages to process (default: 1)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for images (e.g. 0.5 for 50%%)")
    args = parser.parse_args()

    # Load pilot pages
    with open(PILOT_PAGES_JSON) as f:
        pilot_pages = json.load(f)

    pages_to_process = pilot_pages[:args.pages]
    print(f"Will process {len(pages_to_process)} page(s) at scale={args.scale}")

    # Initialize EffOCR with ONNX backend
    print("Initializing EffOCR (ONNX backend)...")
    from efficient_ocr import EffOCR

    hf_repo = "dell-research-harvard/effocr_en"
    effocr = EffOCR(config={
        "Recognizer": {
            "char": {
                "model_backend": "onnx",
                "hf_repo_id": f"{hf_repo}/char_recognizer",
            },
            "word": {
                "model_backend": "onnx",
                "hf_repo_id": f"{hf_repo}/word_recognizer",
            },
        },
        "Localizer": {"model_backend": "onnx", "hf_repo_id": hf_repo},
        "Line": {"model_backend": "onnx", "hf_repo_id": hf_repo},
    })
    print("EffOCR initialized.")

    # Process pages
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in pages_to_process:
        try:
            process_page(effocr, entry, scale=args.scale)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

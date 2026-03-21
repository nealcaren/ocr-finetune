#!/usr/bin/env python3
"""
Multi-resolution OCR evaluation: compare OCR quality at full resolution vs
reduced resolutions (75%, 50%, 30%) to quantify degradation.

Uses AS YOLO pipeline for line detection and EffOCR for recognition.
Full-res OCR output serves as the silver-standard reference.

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/eval_multiresolution.py
"""

import sys
import os
import json
import time
import gc
import tempfile
from pathlib import Path

# Force unbuffered output so progress is visible when redirected
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import cv2
import onnx
import onnxruntime as ort
import torch
from PIL import Image

# Allow imports from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_as_extraction import (
    get_onnx_input_name,
    run_layout_detection,
    run_line_detection,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "effocr"
PILOT_PAGES_JSON = DATA_DIR / "pilot_pages.json"

AS_MODELS_DIR = Path("/Users/nealcaren/Dropbox/american-stories/american_stories_models")
AS_SRC_DIR = Path("/Users/nealcaren/Dropbox/american-stories/AmericanStories/src")

LAYOUT_MODEL_PATH = AS_MODELS_DIR / "layout_model_new.onnx"
LINE_MODEL_PATH = AS_MODELS_DIR / "line_model_new.onnx"
LAYOUT_LABEL_MAP_PATH = AS_SRC_DIR / "label_maps" / "label_map_layout.json"

RESOLUTIONS = [1.0, 0.75, 0.50, 0.30]


# ---------------------------------------------------------------------------
# CER (Character Error Rate) via Levenshtein distance
# ---------------------------------------------------------------------------

def _edit_distance(ref: str, hyp: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1),
            )
    return dp[n]


def cer_from_lines(ref_lines: list, hyp_lines: list) -> float:
    """Compute page-level CER by chunking text into manageable segments.

    Aligns lines by position and computes edit distance per line, then
    aggregates: total_edits / total_ref_chars.  This avoids O(N^2) on
    full 20k-char page strings.
    """
    # Pad the shorter list with empty strings
    max_len = max(len(ref_lines), len(hyp_lines))
    ref_padded = ref_lines + [""] * (max_len - len(ref_lines))
    hyp_padded = hyp_lines + [""] * (max_len - len(hyp_lines))

    total_edits = 0
    total_ref_chars = 0

    for r, h in zip(ref_padded, hyp_padded):
        total_edits += _edit_distance(r, h)
        total_ref_chars += len(r)

    # Account for extra hyp lines beyond ref
    if total_ref_chars == 0:
        return 1.0 if any(hyp_lines) else 0.0

    return total_edits / total_ref_chars


# ---------------------------------------------------------------------------
# Downscaling
# ---------------------------------------------------------------------------

def downscale_cv2(img_cv2, scale):
    """Downscale a cv2 (BGR) image by the given scale factor."""
    if scale >= 1.0:
        return img_cv2
    h, w = img_cv2.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# OCR extraction for a single page at a given resolution
# ---------------------------------------------------------------------------

def extract_text_for_page(
    ca_img,
    layout_session,
    layout_input_name,
    layout_label_map,
    line_session,
    line_input_name,
    effocr,
    scale,
):
    """Run layout -> line detection -> EffOCR on a (possibly downscaled) image.

    Returns (num_lines, list_of_line_texts).
    """
    # Downscale
    img = downscale_cv2(ca_img, scale)
    h, w = img.shape[:2]

    # Layout detection
    layout_crops = run_layout_detection(
        layout_session, layout_input_name, img, layout_label_map
    )

    # Line detection
    all_lines = run_line_detection(line_session, line_input_name, layout_crops)
    num_lines = len(all_lines)

    # OCR each line
    ocr_texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for line_idx, (_layout_idx, _cls, _layout_bbox, _page_bbox, line_crop) in enumerate(all_lines):
            crop_path = Path(tmpdir) / f"line_{line_idx:04d}.png"
            line_crop.save(str(crop_path))

            try:
                results = effocr.infer(str(crop_path))
                # Extract text from results (same logic as run_as_extraction.py)
                text_parts = []
                for bbox_result in results:
                    for k in sorted(bbox_result.preds.keys()):
                        line_data = bbox_result.preds[k]
                        word_preds = line_data.get("word_preds", [])
                        final_puncs = line_data.get("final_puncs", [])
                        parts = []
                        for i, word in enumerate(word_preds):
                            w = word if word else ""
                            if i < len(final_puncs) and final_puncs[i]:
                                w += final_puncs[i]
                            parts.append(w)
                        text_parts.append(" ".join(parts))
                ocr_texts.append(" ".join(text_parts))
            except Exception as e:
                ocr_texts.append("")

    return num_lines, ocr_texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_page_id(entry):
    lccn = entry["lccn"]
    date = entry["date"]
    seq = Path(entry["path"]).stem
    return f"{lccn}_{date}_{seq}"


def main():
    # Load pilot pages -- use last 2 for test
    with open(PILOT_PAGES_JSON) as f:
        pilot_pages = json.load(f)
    test_pages = pilot_pages[-2:]

    print(f"Multi-resolution OCR evaluation")
    print(f"Pages: {len(test_pages)}")
    print(f"Resolutions: {[f'{int(r*100)}%' for r in RESOLUTIONS]}")
    print()

    # Load AS models
    print("Loading layout model...")
    layout_input_name = get_onnx_input_name(str(LAYOUT_MODEL_PATH))
    layout_session = ort.InferenceSession(str(LAYOUT_MODEL_PATH))
    with open(LAYOUT_LABEL_MAP_PATH) as f:
        layout_label_map = {int(k): v for k, v in json.load(f).items()}

    print("Loading line model...")
    line_input_name = get_onnx_input_name(str(LINE_MODEL_PATH))
    line_session = ort.InferenceSession(str(LINE_MODEL_PATH))

    # Load EffOCR
    print("Loading EffOCR...")
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
    print("Models loaded.\n")

    # Collect results: page_id -> {scale: (num_lines, text)}
    all_results = {}

    for entry in test_pages:
        page_id = make_page_id(entry)
        jp2_path = entry["path"]
        print(f"{'='*60}")
        print(f"Page: {page_id}")
        print(f"  JP2: {jp2_path}")

        # Load full-res image once
        ca_img = cv2.imread(jp2_path, cv2.IMREAD_COLOR)
        if ca_img is None:
            print(f"  ERROR: Could not read {jp2_path}")
            continue
        h, w = ca_img.shape[:2]
        print(f"  Full resolution: {w}x{h}")

        page_results = {}
        for scale in RESOLUTIONS:
            pct = int(scale * 100)
            t0 = time.time()
            print(f"\n  {pct}%: running...", end="", flush=True)

            num_lines, line_texts = extract_text_for_page(
                ca_img,
                layout_session, layout_input_name, layout_label_map,
                line_session, line_input_name,
                effocr, scale,
            )
            elapsed = time.time() - t0
            page_results[scale] = (num_lines, line_texts)
            total_chars = sum(len(t) for t in line_texts)
            print(f" {num_lines} lines, {total_chars} chars, {elapsed:.1f}s")
            gc.collect()

        all_results[page_id] = page_results

        # Print per-page CER summary
        ref_lines = page_results[1.0][1]
        print(f"\n  Results for {page_id}:")
        for scale in RESOLUTIONS:
            pct = int(scale * 100)
            num_lines, line_texts = page_results[scale]
            if scale == 1.0:
                print(f"  {pct:>4}%: {num_lines:>4} lines, reference (CER: 0.000)")
            else:
                c = cer_from_lines(ref_lines, line_texts)
                print(f"  {pct:>4}%: {num_lines:>4} lines, CER: {c:.3f}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  {'Resolution':>10} | {'Avg CER':>8} | {'Avg Lines':>10}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for scale in RESOLUTIONS:
        pct = int(scale * 100)
        cers = []
        lines_counts = []
        for page_id, page_results in all_results.items():
            ref_lines = page_results[1.0][1]
            num_lines, line_texts = page_results[scale]
            lines_counts.append(num_lines)
            if scale == 1.0:
                cers.append(0.0)
            else:
                cers.append(cer_from_lines(ref_lines, line_texts))

        avg_cer = sum(cers) / len(cers) if cers else 0.0
        avg_lines = sum(lines_counts) / len(lines_counts) if lines_counts else 0
        print(f"  {pct:>9}% | {avg_cer:>8.3f} | {avg_lines:>10.0f}")


if __name__ == "__main__":
    main()

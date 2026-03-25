#!/usr/bin/env python3
"""
prepare_byt5_data.py — Build training data for ByT5 OCR post-correction.

Groups verified lines by layout region, runs Tesseract on R2 images to get
noisy OCR, then creates (noisy_block, clean_block) pairs with sliding windows.

Usage:
    python scripts/byt5/prepare_byt5_data.py
"""

import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EFFOCR_DIR = DATA_DIR / "effocr"
BYT5_DIR = DATA_DIR / "byt5"
TESSDATA = str(EFFOCR_DIR / "tessdata_best")
EXTRACTIONS_DIR = EFFOCR_DIR / "pre1930_fullres"
TESS_CACHE_PATH = BYT5_DIR / "tesseract_train_cache.json"
HF_REPO = "NealCaren/newspaper-ocr-gold"

WINDOW_SIZE = 10
STRIDE = 5
MIN_BLOCK_LINES = 3


def run_tesseract(img, lang="news_100page"):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        try:
            r = subprocess.run(
                ["tesseract", tmp.name, "stdout", "--psm", "7", "-l", lang,
                 "--tessdata-dir", TESSDATA],
                capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip()
        except:
            return ""
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def dehyphenate_lines(lines):
    """Join lines, removing end-of-line hyphenation."""
    if not lines:
        return ""
    result = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if (stripped.endswith("-")
                and len(stripped) >= 2
                and stripped[-2].isalpha()
                and i + 1 < len(lines)
                and lines[i + 1].lstrip()[:1].islower()):
            result.append(stripped[:-1])
        else:
            result.append(stripped)
            if i < len(lines) - 1:
                result.append(" ")
    return "".join(result)


def load_metadata(split):
    """Load line metadata grouped by (page_id, layout_bbox)."""
    regions = {}  # (page_id, bbox_tuple) -> [line_info, ...]
    split_dir = EXTRACTIONS_DIR / split
    if not split_dir.exists():
        return regions
    for page_dir in sorted(split_dir.iterdir()):
        meta_path = page_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        page_id = meta["page_id"]
        for line_info in meta.get("lines", []):
            bbox = tuple(line_info["layout_bbox"])
            key = (page_id, bbox)
            if key not in regions:
                regions[key] = []
            regions[key].append(line_info)
    # Sort lines within each region by line_index
    for key in regions:
        regions[key].sort(key=lambda x: x["line_index"])
    return regions


def build_hf_lookup(ds, resolution="r2"):
    """Build lookup: (page_id, line_id) -> HF dataset row."""
    lookup = {}
    for row in ds:
        if row["resolution"] != resolution:
            continue
        key = (row["page_id"], row["line_id"])
        lookup[key] = row
    return lookup


def load_tess_cache():
    if TESS_CACHE_PATH.exists():
        return json.loads(TESS_CACHE_PATH.read_text())
    return {}


def save_tess_cache(cache):
    BYT5_DIR.mkdir(parents=True, exist_ok=True)
    TESS_CACHE_PATH.write_text(json.dumps(cache))


def main():
    BYT5_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HF dataset...")
    ds = load_dataset(HF_REPO)

    tess_cache = load_tess_cache()
    print(f"Tesseract cache: {len(tess_cache)} entries")

    for split in ["train", "val", "test"]:
        print(f"\n{'='*50}")
        print(f"Processing {split}")
        print(f"{'='*50}")

        if split not in ds:
            continue

        # Build lookups
        hf_lookup = build_hf_lookup(ds[split], "r2")
        gold_lookup = {}
        for row in ds[split]:
            if row["resolution"] == "full":
                key = (row["page_id"], row["line_id"])
                gold_lookup[key] = row["transcription"].strip()

        print(f"  HF R2 images: {len(hf_lookup)}")
        print(f"  Gold transcriptions: {len(gold_lookup)}")

        # Load metadata for line grouping
        regions = load_metadata(split)
        print(f"  Layout regions: {len(regions)}")

        # Generate Tesseract outputs (with caching)
        new_tess = 0
        for region_key, line_infos in regions.items():
            for li in line_infos:
                page_id, line_id = region_key[0], li["line_id"]
                cache_key = f"{page_id}__{line_id}"
                if cache_key in tess_cache:
                    continue
                hf_key = (page_id, line_id)
                if hf_key not in hf_lookup:
                    continue
                tess_out = run_tesseract(hf_lookup[hf_key]["image"])
                tess_cache[cache_key] = tess_out
                new_tess += 1
                if new_tess % 500 == 0:
                    print(f"    Tesseract: {new_tess} new...")
                    save_tess_cache(tess_cache)

        if new_tess > 0:
            save_tess_cache(tess_cache)
            print(f"  Generated {new_tess} new Tesseract outputs")

        # Build blocks with sliding window
        examples = []
        skipped_regions = 0

        for region_key, line_infos in regions.items():
            page_id = region_key[0]

            # Collect consecutive verified lines
            noisy_lines = []
            gold_lines = []
            for li in line_infos:
                line_id = li["line_id"]
                cache_key = f"{page_id}__{line_id}"
                gold_key = (page_id, line_id)

                tess_out = tess_cache.get(cache_key, "")
                gold_text = gold_lookup.get(gold_key, "")

                if not gold_text or len(gold_text) < 3:
                    # Gap — flush current sequence
                    if len(noisy_lines) >= MIN_BLOCK_LINES:
                        examples.extend(
                            make_windows(noisy_lines, gold_lines, page_id))
                    noisy_lines = []
                    gold_lines = []
                    continue

                noisy_lines.append(tess_out)
                gold_lines.append(gold_text)

            # Flush remaining
            if len(noisy_lines) >= MIN_BLOCK_LINES:
                examples.extend(make_windows(noisy_lines, gold_lines, page_id))
            elif noisy_lines:
                skipped_regions += 1

        print(f"  Examples: {len(examples)}")
        print(f"  Skipped short regions: {skipped_regions}")

        # Save
        out_path = BYT5_DIR / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"  Saved to {out_path}")

        # Stats
        if examples:
            avg_input = sum(len(e["input"]) for e in examples) / len(examples)
            avg_target = sum(len(e["target"]) for e in examples) / len(examples)
            print(f"  Avg input bytes: {avg_input:.0f}")
            print(f"  Avg target bytes: {avg_target:.0f}")


def make_windows(noisy_lines, gold_lines, page_id):
    """Create sliding-window training examples from a sequence of lines."""
    examples = []
    n = len(noisy_lines)

    if n <= WINDOW_SIZE:
        # Single block
        noisy_block = "\n".join(noisy_lines)
        clean_block = dehyphenate_lines(gold_lines)
        if noisy_block.strip() and clean_block.strip():
            examples.append({
                "input": f"correct: {noisy_block}",
                "target": clean_block,
                "page_id": page_id,
                "num_lines": n,
            })
    else:
        # Sliding window
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            noisy_block = "\n".join(noisy_lines[start:end])
            clean_block = dehyphenate_lines(gold_lines[start:end])
            if noisy_block.strip() and clean_block.strip():
                examples.append({
                    "input": f"correct: {noisy_block}",
                    "target": clean_block,
                    "page_id": page_id,
                    "num_lines": WINDOW_SIZE,
                })
        # Last window if we didn't cover the end
        if (n - WINDOW_SIZE) % STRIDE != 0:
            noisy_block = "\n".join(noisy_lines[-WINDOW_SIZE:])
            clean_block = dehyphenate_lines(gold_lines[-WINDOW_SIZE:])
            if noisy_block.strip() and clean_block.strip():
                examples.append({
                    "input": f"correct: {noisy_block}",
                    "target": clean_block,
                    "page_id": page_id,
                    "num_lines": WINDOW_SIZE,
                })

    return examples


if __name__ == "__main__":
    main()

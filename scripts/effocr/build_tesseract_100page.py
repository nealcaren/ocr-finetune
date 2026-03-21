"""Build Tesseract training data from LLM-verified 100-page line crops.

Reads the verified JSONL and creates PNG + .gt.txt pairs for Tesseract fine-tuning.

Includes:
  A) Positive examples: clean/partial/degraded lines, confidence >= 0.8, text >= 3 chars
  B) LLM-flagged negatives: illegible/not_text lines -> blank .gt.txt
  C) Thin-line negatives: crops with height < 15px (rule lines) -> blank .gt.txt
  D) Mined noise strips: random horizontal strips from page margins/gaps -> blank .gt.txt

Target: ~5-10% blank examples in training set.

Output structure:
  data/effocr/tesseract_100page_v2/{train,val,test}/{page_id}__{line_id}.{png,gt.txt}

Usage:
    python scripts/effocr/build_tesseract_100page.py
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"
VERIFIED_PATH = DATA_DIR / "100page_verified.jsonl"
EXTRACTIONS_DIR = DATA_DIR / "100page_extractions"
OUTPUT_DIR = DATA_DIR / "tesseract_100page_v2"

# Filter criteria for positive examples
MIN_CONFIDENCE = 0.8
MIN_TEXT_LENGTH = 3
POSITIVE_FLAGS = {"clean", "degraded", "partial"}

# Blank example criteria
BLANK_FLAGS = {"illegible", "not_text"}
THIN_LINE_MAX_HEIGHT = 15  # pixels

# Noise mining settings
TARGET_BLANK_FRACTION = 0.07  # ~7% blanks
NOISE_STRIP_HEIGHT_RANGE = (30, 60)  # typical text-line height range
NOISE_STRIP_MIN_WIDTH = 200

random.seed(42)


def load_entries():
    """Load all verified lines from JSONL."""
    entries = []
    for line in VERIFIED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def filter_positives(entries):
    """Filter entries for positive training examples."""
    return [
        e for e in entries
        if e["flag"] in POSITIVE_FLAGS
        and e["confidence"] >= MIN_CONFIDENCE
        and len(e["transcription"].strip()) >= MIN_TEXT_LENGTH
    ]


def get_blank_flagged(entries):
    """Get LLM-flagged negatives (illegible/not_text)."""
    return [e for e in entries if e["flag"] in BLANK_FLAGS]


def get_thin_lines(entries):
    """Find thin line crops (height < 15px) that are likely rule lines, not text.

    Cross-references metadata to get crop heights, then returns entries
    whose crops should be treated as blank.
    """
    # Build a lookup of line heights from metadata
    line_heights = {}  # (split, page_id, line_id) -> height
    for split in ["train", "val", "test"]:
        split_dir = EXTRACTIONS_DIR / split
        if not split_dir.exists():
            continue
        for page_dir in split_dir.iterdir():
            meta_path = page_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.load(open(meta_path))
            page_id = meta["page_id"]
            for line_info in meta.get("lines", []):
                h = line_info["crop_size"][1]
                line_heights[(split, page_id, line_info["line_id"])] = h

    thin = []
    for e in entries:
        key = (e["split"], e["page_id"], e["line_id"])
        h = line_heights.get(key, 999)
        if h < THIN_LINE_MAX_HEIGHT:
            # Only include if it wasn't already a positive (avoid double-counting)
            if e["flag"] in POSITIVE_FLAGS and e["confidence"] >= MIN_CONFIDENCE:
                thin.append(e)
    return thin


def mine_noise_strips(split, target_count):
    """Mine random noise strips from gaps between text regions on page images.

    For each page, identifies vertical gaps between text-region bounding boxes
    and crops horizontal strips from those gaps.
    """
    if Image is None:
        print("  WARNING: Pillow not available, skipping noise mining")
        return []

    split_dir = EXTRACTIONS_DIR / split
    if not split_dir.exists():
        return []

    # Collect page metadata
    pages = []
    for page_dir in sorted(split_dir.iterdir()):
        meta_path = page_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        pages.append((page_dir, meta))

    if not pages:
        return []

    strips_per_page = max(1, target_count // len(pages) + 1)
    results = []

    for page_dir, meta in pages:
        jp2_path = Path(meta.get("jp2_path", ""))
        if not jp2_path.exists():
            continue

        img_w, img_h = meta["image_size"]

        # Collect all text-region bounding boxes
        region_bboxes = []
        for r in meta.get("selected_regions", []):
            region_bboxes.append(r["bbox"])  # [x1, y1, x2, y2]

        # Also collect line-level bboxes for finer gap detection
        line_bboxes = []
        for line_info in meta.get("lines", []):
            line_bboxes.append(line_info["page_bbox"])

        # Find vertical gaps: sort all bboxes by y1, find spaces between y2 and next y1
        all_bboxes = region_bboxes + line_bboxes
        if not all_bboxes:
            continue

        # Sort by top edge
        all_bboxes.sort(key=lambda b: b[1])

        # Find gaps
        gaps = []
        # Gap at top of page
        if all_bboxes[0][1] > NOISE_STRIP_HEIGHT_RANGE[1] * 2:
            gaps.append((0, 0, img_w, all_bboxes[0][1]))

        # Gaps between regions
        for i in range(len(all_bboxes) - 1):
            y2_current = all_bboxes[i][3]
            y1_next = all_bboxes[i + 1][1]
            gap_height = y1_next - y2_current
            if gap_height >= NOISE_STRIP_HEIGHT_RANGE[0]:
                gaps.append((0, y2_current, img_w, y1_next))

        # Gap at bottom of page
        if img_h - all_bboxes[-1][3] > NOISE_STRIP_HEIGHT_RANGE[1] * 2:
            gaps.append((0, all_bboxes[-1][3], img_w, img_h))

        if not gaps:
            continue

        # Try to open the image
        try:
            page_img = Image.open(jp2_path)
        except Exception as e:
            print(f"  WARNING: cannot open {jp2_path}: {e}")
            continue

        page_id = meta["page_id"]
        strips_mined = 0

        random.shuffle(gaps)
        for gap in gaps:
            if strips_mined >= strips_per_page:
                break

            gx1, gy1, gx2, gy2 = gap
            gap_h = gy2 - gy1
            strip_h = random.randint(*NOISE_STRIP_HEIGHT_RANGE)
            if gap_h < strip_h:
                strip_h = gap_h

            # Random y position within gap
            max_y = gy2 - strip_h
            if max_y <= gy1:
                continue
            y = random.randint(gy1, max_y)

            # Random x crop (partial width for variety)
            crop_w = random.randint(min(NOISE_STRIP_MIN_WIDTH, gx2 - gx1), gx2 - gx1)
            x = random.randint(gx1, max(gx1, gx2 - crop_w))

            try:
                strip = page_img.crop((x, y, x + crop_w, y + strip_h))
                if strip.size[0] < NOISE_STRIP_MIN_WIDTH:
                    continue
            except Exception:
                continue

            noise_id = f"noise_{strips_mined:04d}"
            results.append({
                "page_id": page_id,
                "line_id": noise_id,
                "split": split,
                "image": strip,
                "source": "mined_gap",
            })
            strips_mined += 1

        page_img.close()

    # Trim to target count
    if len(results) > target_count:
        random.shuffle(results)
        results = results[:target_count]

    return results


def main():
    if not VERIFIED_PATH.exists():
        print(f"ERROR: {VERIFIED_PATH} not found. Run verify_100_pages.py first.")
        return

    # Load all verified lines
    entries = load_entries()
    print(f"Loaded {len(entries)} verified lines")

    # --- A) Positive examples ---
    positives = filter_positives(entries)
    print(f"\nPositive examples (flag in {POSITIVE_FLAGS}, conf >= {MIN_CONFIDENCE}, "
          f"len >= {MIN_TEXT_LENGTH}): {len(positives)}")

    # Show rejection stats
    rejected_flags = Counter()
    for e in entries:
        if e not in positives:
            rejected_flags[e["flag"]] += 1
    if rejected_flags:
        print(f"  Rejected by flag: {dict(rejected_flags)}")
    low_conf = sum(1 for e in entries if e["flag"] in POSITIVE_FLAGS and e["confidence"] < MIN_CONFIDENCE)
    short = sum(1 for e in entries if e["flag"] in POSITIVE_FLAGS and e["confidence"] >= MIN_CONFIDENCE
                and len(e["transcription"].strip()) < MIN_TEXT_LENGTH)
    print(f"  Rejected: low confidence={low_conf}, too short={short}")

    # Count positives by split
    pos_by_split = Counter(e["split"] for e in positives)
    print(f"  By split: {dict(pos_by_split)}")

    # --- B) LLM-flagged negatives ---
    blank_flagged = get_blank_flagged(entries)
    print(f"\nLLM-flagged blanks (illegible/not_text): {len(blank_flagged)}")

    # --- C) Thin-line negatives ---
    thin_lines = get_thin_lines(entries)
    # Remove any that are already in blank_flagged
    blank_flagged_keys = {(e["split"], e["page_id"], e["line_id"]) for e in blank_flagged}
    thin_lines = [e for e in thin_lines
                  if (e["split"], e["page_id"], e["line_id"]) not in blank_flagged_keys]
    print(f"Thin-line blanks (height < {THIN_LINE_MAX_HEIGHT}px): {len(thin_lines)}")

    # --- Calculate how many mined noise strips we need ---
    existing_blanks = len(blank_flagged) + len(thin_lines)
    train_positives = pos_by_split.get("train", 0)
    target_train_blanks = int(train_positives * TARGET_BLANK_FRACTION)
    # Count existing blanks in training split
    existing_train_blanks = (
        sum(1 for e in blank_flagged if e["split"] == "train")
        + sum(1 for e in thin_lines if e["split"] == "train")
    )
    needed_train_noise = max(0, target_train_blanks - existing_train_blanks)

    # Also mine some for val/test
    val_positives = pos_by_split.get("val", 0)
    test_positives = pos_by_split.get("test", 0)
    target_val_blanks = int(val_positives * TARGET_BLANK_FRACTION)
    target_test_blanks = int(test_positives * TARGET_BLANK_FRACTION)
    existing_val_blanks = (
        sum(1 for e in blank_flagged if e["split"] == "val")
        + sum(1 for e in thin_lines if e["split"] == "val")
    )
    existing_test_blanks = (
        sum(1 for e in blank_flagged if e["split"] == "test")
        + sum(1 for e in thin_lines if e["split"] == "test")
    )
    needed_val_noise = max(0, target_val_blanks - existing_val_blanks)
    needed_test_noise = max(0, target_test_blanks - existing_test_blanks)

    print(f"\nNoise mining targets: train={needed_train_noise}, val={needed_val_noise}, test={needed_test_noise}")

    # --- D) Mine noise strips ---
    mined_noise = {}
    for split, needed in [("train", needed_train_noise), ("val", needed_val_noise), ("test", needed_test_noise)]:
        if needed > 0:
            print(f"  Mining {needed} noise strips for {split}...")
            mined_noise[split] = mine_noise_strips(split, needed)
            print(f"  Got {len(mined_noise[split])} noise strips for {split}")
        else:
            mined_noise[split] = []

    # --- Write output ---
    # Clear old output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    counts = {"train": {"positive": 0, "blank": 0}, "val": {"positive": 0, "blank": 0},
              "test": {"positive": 0, "blank": 0}}

    # Multi-resolution scales: each positive example also gets downscaled copies
    # so Tesseract learns to read text at reduced resolutions (target: ~2800px page height)
    DOWNSCALE_FACTORS = [0.5, 0.35]  # 50% and 35% of full res

    # A) Write positive examples (full-res + downscaled)
    for entry in positives:
        split = entry["split"]
        page_id = entry["page_id"]
        line_id = entry["line_id"]
        text = entry["transcription"].strip()
        crop_path = Path(entry["crop_path"])

        if not crop_path.exists():
            print(f"  WARNING: missing {crop_path}")
            continue

        out_name = f"{page_id}__{line_id}"
        out_dir = OUTPUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Full-res copy
        shutil.copy2(crop_path, out_dir / f"{out_name}.png")
        (out_dir / f"{out_name}.gt.txt").write_text(text, encoding="utf-8")
        counts[split]["positive"] += 1

        # Downscaled copies (same ground truth text)
        if Image is not None:
            try:
                img = Image.open(crop_path)
                for scale in DOWNSCALE_FACTORS:
                    new_w = max(1, int(img.width * scale))
                    new_h = max(1, int(img.height * scale))
                    # Skip if too small to be useful
                    if new_w < 50 or new_h < 8:
                        continue
                    scaled = img.resize((new_w, new_h), Image.LANCZOS)
                    scale_name = f"{int(scale * 100)}pct"
                    scaled.save(out_dir / f"{out_name}_{scale_name}.png")
                    (out_dir / f"{out_name}_{scale_name}.gt.txt").write_text(text, encoding="utf-8")
                    counts[split]["positive"] += 1
                img.close()
            except Exception as e:
                pass  # Skip downscaling on error, full-res still saved

    # B) Write LLM-flagged blanks
    for entry in blank_flagged:
        split = entry["split"]
        page_id = entry["page_id"]
        line_id = entry["line_id"]
        crop_path = Path(entry["crop_path"])

        if not crop_path.exists():
            print(f"  WARNING: missing blank crop {crop_path}")
            continue

        out_name = f"{page_id}__{line_id}"
        out_dir = OUTPUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(crop_path, out_dir / f"{out_name}.png")
        (out_dir / f"{out_name}.gt.txt").write_text("", encoding="utf-8")
        counts[split]["blank"] += 1

    # C) Write thin-line blanks (override their text with blank)
    for entry in thin_lines:
        split = entry["split"]
        page_id = entry["page_id"]
        line_id = entry["line_id"]
        crop_path = Path(entry["crop_path"])

        if not crop_path.exists():
            continue

        out_name = f"{page_id}__{line_id}"
        out_dir = OUTPUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Overwrite the positive .gt.txt with blank
        shutil.copy2(crop_path, out_dir / f"{out_name}.png")
        (out_dir / f"{out_name}.gt.txt").write_text("", encoding="utf-8")
        # Adjust counts: was counted as positive, now blank
        if out_name in [f"{e['page_id']}__{e['line_id']}" for e in positives
                        if e["split"] == split]:
            counts[split]["positive"] -= 1
        counts[split]["blank"] += 1

    # D) Write mined noise strips
    for split, strips in mined_noise.items():
        out_dir = OUTPUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for strip_info in strips:
            out_name = f"{strip_info['page_id']}__{strip_info['line_id']}"
            strip_info["image"].save(out_dir / f"{out_name}.png")
            (out_dir / f"{out_name}.gt.txt").write_text("", encoding="utf-8")
            counts[split]["blank"] += 1

    # --- Summary ---
    print(f"\nTesseract training data created at: {OUTPUT_DIR}")
    total_pos = 0
    total_blank = 0
    for split in ["train", "val", "test"]:
        p = counts[split]["positive"]
        b = counts[split]["blank"]
        total = p + b
        pct = (b / total * 100) if total > 0 else 0
        print(f"  {split}: {total} pairs ({p} positive, {b} blank [{pct:.1f}%])")
        total_pos += p
        total_blank += b
    grand = total_pos + total_blank
    pct = (total_blank / grand * 100) if grand > 0 else 0
    print(f"  TOTAL: {grand} pairs ({total_pos} positive, {total_blank} blank [{pct:.1f}%])")


if __name__ == "__main__":
    main()

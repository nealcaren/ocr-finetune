#!/usr/bin/env python3
"""
build_gold_regions.py — Build ByT5 training data from the actual pipeline.

1. PPStructureV3 layout detection on pre-1930 pages
2. Merge adjacent blocks (matching newspaper_ocr.py logic)
3. Tesseract --psm 6 on each region crop → noisy input
4. Qwen3-VL 235B on each region crop → gold standard target
5. Save as ByT5 training pairs

Usage:
    python scripts/byt5/build_gold_regions.py
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from PIL import Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BYT5_DIR = DATA_DIR / "byt5"
EFFOCR_DIR = DATA_DIR / "effocr"
SAMPLE_JSON = EFFOCR_DIR / "pre1930_sample.json"
TESSDATA = str(EFFOCR_DIR / "tessdata_best")
OUTPUT_DIR = BYT5_DIR / "pipeline_regions"
GOLD_JSONL = BYT5_DIR / "pipeline_gold.jsonl"

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
QWEN_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

OCR_LABELS = {"text", "paragraph_title", "doc_title", "figure_title"}

QWEN_PROMPT = """You are examining a cropped image of a text region from a historical newspaper (pre-1930).

Transcribe the COMPLETE text as a flowing paragraph:
1. Preserve original spelling, capitalization, and punctuation exactly.
2. Join lines naturally — remove end-of-line hyphens where words were split.
3. Keep real hyphens in compound words (e.g., "well-known").
4. Preserve paragraph breaks if visible.

Respond with ONLY a JSON object:
{"transcription": "the flowing text...", "confidence": 0.95}"""


def merge_adjacent_blocks(blocks, x_overlap_thresh=0.5, y_gap_max=30):
    """Merge consecutive same-column blocks. Matches newspaper_ocr.py logic."""
    if not blocks:
        return blocks
    merged, current = [], None
    for block in blocks:
        label = block["block_label"]
        x1, y1, x2, y2 = block["block_bbox"]
        if label in ("doc_title", "paragraph_title"):
            if current:
                merged.append(current)
            merged.append({"block_label": label, "block_bbox": [x1, y1, x2, y2],
                           "block_order": block["block_order"]})
            current = None
            continue
        if current is None:
            current = {"block_label": "text", "block_bbox": [x1, y1, x2, y2],
                       "block_order": block["block_order"]}
            continue
        cx1, cy1, cx2, cy2 = current["block_bbox"]
        overlap = max(0, min(cx2, x2) - max(cx1, x1))
        ratio = overlap / min(cx2 - cx1, x2 - x1) if min(cx2 - cx1, x2 - x1) > 0 else 0
        y_gap = y1 - cy2
        if ratio >= x_overlap_thresh and 0 <= y_gap <= y_gap_max:
            current["block_bbox"] = [min(cx1, x1), cy1, max(cx2, x2), max(cy2, y2)]
        else:
            merged.append(current)
            current = {"block_label": "text", "block_bbox": [x1, y1, x2, y2],
                       "block_order": block["block_order"]}
    if current:
        merged.append(current)
    return merged


def run_tesseract_region(image):
    """Run Tesseract in block mode on a region crop."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        try:
            r = subprocess.run(
                ["tesseract", tmp.name, "stdout", "--psm", "6", "-l", "news_100page",
                 "--tessdata-dir", TESSDATA],
                capture_output=True, text=True, timeout=30
            )
            return r.stdout.strip()
        except:
            return ""
        finally:
            Path(tmp.name).unlink(missing_ok=True)


async def qwen_ocr(client, semaphore, image):
    """Send region crop to Qwen3-VL for gold transcription."""
    async with semaphore:
        buf = io.BytesIO()
        # Resize if too large
        w, h = image.size
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": QWEN_PROMPT},
            ]}],
            "temperature": 0.1,
            "max_tokens": 4000,
        }

        for attempt in range(5):
            try:
                resp = await client.post(
                    API_URL, json=payload,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    timeout=180.0,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                if "<think>" in content:
                    content = content.split("</think>")[-1].strip()
                parsed = json.loads(content)
                return parsed.get("transcription", ""), parsed.get("confidence", 0.0)
            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ReadError,
                    httpx.ConnectError, json.JSONDecodeError, KeyError, IndexError) as e:
                if attempt == 4:
                    return "", 0.0
                await asyncio.sleep(2 ** attempt)
        return "", 0.0


async def process_page(page_entry, split, layout_engine, client, semaphore):
    """Process one page: layout → regions → Tesseract + Qwen → training pairs."""
    jp2_path = page_entry["path"]
    page_id = f"{page_entry['lccn']}_{page_entry['date']}_{Path(jp2_path).stem}"

    # Check resume
    page_dir = OUTPUT_DIR / split / page_id
    meta_path = page_dir / "metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        return meta.get("regions", [])

    # Load image, downscale to R2 (matching deployment resolution)
    full_image = Image.open(jp2_path).convert("RGB")
    r2_w = int(full_image.width * 0.35)
    r2_h = int(full_image.height * 0.35)
    full_image = full_image.resize((r2_w, r2_h), Image.LANCZOS)
    img_w, img_h = full_image.size

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        full_image.save(tmp.name, quality=85)
        tmp_path = tmp.name

    # Layout detection (PP-DocLayout, same format as ocr_pages.py)
    boxes_raw = []
    for result in layout_engine.predict(tmp_path):
        boxes_raw = result["boxes"]
    Path(tmp_path).unlink(missing_ok=True)

    text_blocks = []
    for i, b in enumerate(boxes_raw):
        if b["score"] > 0.5 and b["label"] in OCR_LABELS:
            text_blocks.append({
                "block_label": b["label"],
                "block_bbox": [int(c) for c in b["coordinate"]],
                "block_order": i,
            })
    # Sort by y position (top to bottom reading order)
    text_blocks.sort(key=lambda b: b["block_bbox"][1])
    merged = merge_adjacent_blocks(text_blocks)

    # Process each region
    regions = []
    for i, block in enumerate(merged):
        x1, y1, x2, y2 = [int(c) for c in block["block_bbox"]]
        region_crop = full_image.crop((x1, y1, x2, y2))
        rw, rh = region_crop.size

        # Tesseract on the region
        tess_text = run_tesseract_region(region_crop)

        # Qwen gold transcription
        gold_text, confidence = await qwen_ocr(client, semaphore, region_crop)

        # Save region crop
        page_dir.mkdir(parents=True, exist_ok=True)
        crop_path = page_dir / f"region_{i:02d}.png"
        region_crop.save(str(crop_path))

        regions.append({
            "region_id": f"region_{i:02d}",
            "label": block["block_label"],
            "bbox": [x1, y1, x2, y2],
            "size": [rw, rh],
            "tesseract": tess_text,
            "gold": gold_text,
            "confidence": confidence,
        })

    full_image.close()

    # Save metadata
    meta = {
        "page_id": page_id,
        "split": split,
        "jp2_path": jp2_path,
        "num_regions": len(regions),
        "regions": regions,
    }
    page_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return regions


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BYT5_DIR.mkdir(parents=True, exist_ok=True)

    # Load sample pages
    with open(SAMPLE_JSON) as f:
        splits = json.load(f)
    total = sum(len(v) for v in splits.values())
    print(f"Loaded {total} pages")

    # Init layout engine (PP-DocLayout, same as ocr_pages.py)
    print("Loading PP-DocLayout_plus-L...")
    from paddlex import create_model
    engine = create_model("PP-DocLayout_plus-L")
    print("Ready.")

    semaphore = asyncio.Semaphore(3)
    processed = 0

    async with httpx.AsyncClient() as client:
        with open(GOLD_JSONL, "a") as gold_f:
            for split_name in ["train", "val", "test"]:
                pages = splits.get(split_name, [])
                print(f"\n{'='*50}")
                print(f"{split_name}: {len(pages)} pages")

                for entry in pages:
                    page_id = f"{entry['lccn']}_{entry['date']}_{Path(entry['path']).stem}"
                    processed += 1
                    print(f"  [{processed}/{total}] {page_id}", end="", flush=True)
                    t0 = time.time()

                    regions = await process_page(entry, split_name, engine, client, semaphore)

                    # Write training pairs
                    for r in regions:
                        if not r["gold"] or r["confidence"] < 0.8:
                            continue
                        if not r["tesseract"]:
                            continue
                        # Check ByT5 size limits
                        input_text = f"correct: {r['tesseract']}"
                        if len(input_text.encode()) > 1024 or len(r["gold"].encode()) > 1024:
                            continue
                        gold_f.write(json.dumps({
                            "split": split_name,
                            "page_id": page_id,
                            "region_id": r["region_id"],
                            "input": input_text,
                            "target": r["gold"],
                            "label": r["label"],
                            "confidence": r["confidence"],
                        }) + "\n")
                        gold_f.flush()

                    elapsed = time.time() - t0
                    n_good = sum(1 for r in regions if r.get("gold") and r["confidence"] >= 0.8)
                    print(f" → {len(regions)} regions, {n_good} good, {elapsed:.0f}s")

    # Split into train/val/test JSON files
    print("\nSplitting into JSON files...")
    by_split = {"train": [], "val": [], "test": []}
    with open(GOLD_JSONL) as f:
        for line in f:
            d = json.loads(line)
            by_split[d["split"]].append({
                "input": d["input"],
                "target": d["target"],
                "page_id": d["page_id"],
                "region_id": d["region_id"],
            })

    for split, examples in by_split.items():
        out = BYT5_DIR / f"{split}_pipeline.json"
        with open(out, "w") as f:
            json.dump(examples, f, indent=2)
        if examples:
            avg_in = sum(len(e["input"]) for e in examples) / len(examples)
            avg_tgt = sum(len(e["target"]) for e in examples) / len(examples)
            print(f"  {split}: {len(examples)} examples, avg input {avg_in:.0f}b, avg target {avg_tgt:.0f}b")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

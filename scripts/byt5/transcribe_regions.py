#!/usr/bin/env python3
"""
transcribe_regions.py — Send full layout regions to Qwen3-VL 235B for
block-level transcription to create gold-standard flowing text.

Crops layout regions from full-res JP2s and sends them to Qwen3-VL via
OpenRouter. Returns flowing paragraph text with natural dehyphenation.

Usage:
    python scripts/byt5/transcribe_regions.py
"""

import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import httpx
from PIL import Image

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT = 3

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EFFOCR_DIR = DATA_DIR / "effocr"
EXTRACTIONS_DIR = EFFOCR_DIR / "pre1930_fullres"
OUTPUT_PATH = DATA_DIR / "byt5" / "region_transcriptions.jsonl"

PROMPT = """You are examining a cropped image of a text region from a historical newspaper (pre-1930). The image contains multiple lines of text from a single article or section.

Please transcribe the COMPLETE text as a flowing paragraph:
1. Preserve original spelling, capitalization, and punctuation exactly as printed.
2. Join lines naturally — remove end-of-line hyphens where a word was split across lines (e.g., "news-\\npaper" becomes "newspaper"), but keep real hyphens in compound words (e.g., "well-known").
3. Preserve paragraph breaks if visible (use a blank line).
4. If text is cut off at the edges, transcribe what is visible.
5. Skip any non-text elements (illustrations, decorative rules).

Rate your confidence (0.0-1.0) in the transcription.

Respond with ONLY a JSON object (no markdown, no explanation):
{"transcription": "the flowing paragraph text...", "confidence": 0.95, "note": "optional brief note"}"""


def crop_region(jp2_path, bbox):
    """Crop a layout region from a JP2 image."""
    img = Image.open(jp2_path)
    x0, y0, x1, y1 = bbox
    crop = img.crop((x0, y0, x1, y1))
    img.close()
    return crop


def image_to_base64(img):
    """Convert PIL Image to base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def collect_regions():
    """Collect all layout regions with metadata."""
    regions = []
    for split in ("train", "val", "test"):
        split_dir = EXTRACTIONS_DIR / split
        if not split_dir.exists():
            continue
        for page_dir in sorted(split_dir.iterdir()):
            meta_path = page_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.load(open(meta_path))
            jp2_path = meta.get("jp2_path", "")
            if not Path(jp2_path).exists():
                continue
            page_id = meta["page_id"]

            for i, region in enumerate(meta.get("selected_regions", [])):
                region_id = f"region_{i:02d}"
                # Collect line IDs that belong to this region
                region_bbox = tuple(region["bbox"])
                line_ids = [
                    li["line_id"] for li in meta.get("lines", [])
                    if tuple(li["layout_bbox"]) == region_bbox
                ]
                regions.append({
                    "split": split,
                    "page_id": page_id,
                    "region_id": region_id,
                    "region_class": region["class"],
                    "bbox": region["bbox"],
                    "num_lines": region["num_lines"],
                    "line_ids": line_ids,
                    "jp2_path": jp2_path,
                })
    return regions


def load_done_keys():
    """Load already-processed region keys."""
    done = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                done.add(f"{entry['split']}__{entry['page_id']}__{entry['region_id']}")
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def transcribe_region(client, semaphore, region):
    """Send a region crop to Qwen3-VL for transcription."""
    async with semaphore:
        # Crop the region
        try:
            crop = crop_region(region["jp2_path"], region["bbox"])
        except Exception as e:
            return {**region, "transcription": "", "confidence": 0.0,
                    "note": f"crop error: {e}", "status": "error"}

        # Resize if too large (API limit)
        max_dim = 2048
        w, h = crop.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img_b64 = image_to_base64(crop)
        crop.close()

        payload = {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 4000,
        }

        for attempt in range(5):
            try:
                resp = await client.post(
                    API_URL, json=payload,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=180.0,
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1) + (attempt * 2)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"].strip()

                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                if "<think>" in content:
                    content = content.split("</think>")[-1].strip()

                parsed = json.loads(content)
                return {
                    "split": region["split"],
                    "page_id": region["page_id"],
                    "region_id": region["region_id"],
                    "region_class": region["region_class"],
                    "bbox": region["bbox"],
                    "num_lines": region["num_lines"],
                    "line_ids": region["line_ids"],
                    "transcription": parsed.get("transcription", ""),
                    "confidence": parsed.get("confidence", 0.0),
                    "note": parsed.get("note", ""),
                    "status": "ok",
                }

            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ReadError,
                    httpx.ConnectError, httpx.RemoteProtocolError,
                    json.JSONDecodeError, KeyError, IndexError) as e:
                if attempt == 4:
                    return {**region, "transcription": "", "confidence": 0.0,
                            "note": str(e), "status": "error"}
                await asyncio.sleep(2 ** attempt)

        return {**region, "transcription": "", "confidence": 0.0,
                "note": "max retries", "status": "error"}


async def main():
    regions = collect_regions()
    print(f"Found {len(regions)} layout regions")
    for split in ("train", "val", "test"):
        count = sum(1 for r in regions if r["split"] == split)
        print(f"  {split}: {count}")

    done_keys = load_done_keys()
    remaining = [
        r for r in regions
        if f"{r['split']}__{r['page_id']}__{r['region_id']}" not in done_keys
    ]
    print(f"\nAlready processed: {len(done_keys)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All regions already processed!")
        return

    print(f"\nSending to {MODEL} via OpenRouter (max {MAX_CONCURRENT} concurrent)...")
    start_time = time.time()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        tasks = [transcribe_region(client, semaphore, r) for r in remaining]
        completed = 0
        errors = 0

        with open(OUTPUT_PATH, "a") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                f.write(json.dumps(result) + "\n")
                f.flush()
                completed += 1
                if result.get("status") == "error":
                    errors += 1

                if completed % 10 == 0 or completed == len(remaining):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{len(done_keys)+completed}/{len(regions)}] "
                          f"{completed} done in {elapsed/60:.1f}min "
                          f"({rate:.2f}/s) errors={errors}")

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} regions in {elapsed/60:.1f} min, {errors} errors")


if __name__ == "__main__":
    asyncio.run(main())

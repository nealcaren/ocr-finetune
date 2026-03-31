#!/usr/bin/env python3
"""
eval_lightonocr.py — Evaluate LightOnOCR-2-1B on region images.

Usage:
    python scripts/lightonocr/eval_lightonocr.py --limit 100
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

REPO = Path(__file__).resolve().parent.parent.parent


def cer(ref, hyp):
    if not ref:
        return 1.0 if hyp else 0.0
    m, n = len(ref), len(hyp)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, n + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + c)
    return d[n] / m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="lightonai/LightOnOCR-2-1B-base")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {args.model}")

    print("Loading model...")
    processor = LightOnOcrProcessor.from_pretrained(args.model)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    print("Loaded.")

    # Load gold
    gold = {}
    for line in open(REPO / "data/byt5/pipeline_gold.jsonl"):
        d = json.loads(line)
        if d["split"] == "test":
            gold[f"{d['page_id']}/{d['region_id']}"] = d["target"]

    # Test on regions
    regions_dir = REPO / "data/byt5/pipeline_regions/test"
    cers = []
    count = 0
    t0 = time.time()

    for page_dir in sorted(regions_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        meta = json.load(open(page_dir / "metadata.json"))
        for r in meta["regions"]:
            key = f"{meta['page_id']}/{r['region_id']}"
            if key not in gold:
                continue
            img_path = page_dir / f"{r['region_id']}.png"
            if not img_path.exists():
                continue

            conversation = [{"role": "user", "content": [{"type": "image", "url": str(img_path)}]}]
            inputs = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

            pred = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            ref = gold[key]
            c = cer(ref, pred)
            cers.append(c)
            count += 1

            if count <= 10:
                print(f"Gold: {ref[:70]}")
                print(f"Pred: {pred[:70]}")
                print(f"CER:  {c * 100:.1f}%")
                print()

            if count % 50 == 0:
                elapsed = time.time() - t0
                avg = sum(cers) / len(cers) * 100
                print(f"  [{count}] CER={avg:.2f}%, {elapsed:.0f}s, {count/elapsed:.1f} regions/s")

            if count >= args.limit:
                break
        if count >= args.limit:
            break

    avg = sum(cers) / len(cers) * 100
    med = sorted(cers)[len(cers) // 2] * 100
    elapsed = time.time() - t0
    print(f"\nLightOnOCR on R2 regions (n={len(cers)}): CER={avg:.2f}%, median={med:.2f}%")
    print(f"Time: {elapsed:.0f}s ({count/elapsed:.1f} regions/s)")


if __name__ == "__main__":
    main()

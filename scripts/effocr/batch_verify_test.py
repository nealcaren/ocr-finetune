"""Batch test of LLM line verification via OpenRouter.

Sends a diverse sample of line crops to Qwen3-VL for:
1. Blind transcription (no OCR text shown first)
2. Quality assessment and flagging
3. Comparison against EffOCR silver-standard

Usage:
    python scripts/effocr/batch_verify_test.py [--n 50]
"""

import asyncio
import base64
import json
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT = 3

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"
OUTPUT_PATH = DATA_DIR / "batch_verify_test.jsonl"

PROMPT = """You are examining a cropped image of a single line of text from a historical newspaper (1840s-1930s era). The image was extracted by an automated line detector.

Please:
1. Transcribe exactly what you see. Preserve original spelling, capitalization, and punctuation.
2. Rate your confidence (0.0-1.0) in your transcription.
3. Flag any issues from this list:
   - "clean" — normal readable text, no issues
   - "degraded" — text is readable but print quality is poor (faded, smudged, broken characters)
   - "partial" — line appears cut off or only partially captured
   - "not_text" — image contains non-text content (illustration, decoration, rule line, etc.)
   - "illegible" — cannot reliably read most of the text
   - "table" — structured/tabular content rather than running text
   - "header" — page header, page number, or masthead (not article content)
   - "multi_line" — image contains more than one line of text
   - "foreign" — non-English text

Respond with ONLY a JSON object (no markdown, no explanation):
{"transcription": "exact text you see", "confidence": 0.95, "flag": "clean", "note": "optional brief note about anything unusual"}"""


def image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def select_sample(n: int) -> list[dict]:
    """Select a diverse sample of line crops across pages."""
    labels = {}
    for line in (DATA_DIR / "line_labels.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        labels[str(Path(entry["crop_path"]).resolve())] = entry["text"]

    # Collect all line crops grouped by page
    extractions_dir = DATA_DIR / "as_extractions"
    by_page = {}
    for page_dir in sorted(extractions_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        lines_dir = page_dir / "lines"
        if not lines_dir.exists():
            continue
        crops = sorted(lines_dir.glob("line_*.png"))
        if crops:
            by_page[page_dir.name] = crops

    # When n=0 (--all), include every crop. Otherwise sample evenly.
    rng = random.Random(42)
    selected = []
    pages = sorted(by_page.keys())

    if n == 0:
        # Include ALL crops
        for page_id in pages:
            for crop in by_page[page_id]:
                effocr_text = labels.get(str(crop.resolve()), "")
                selected.append({
                    "page_id": page_id,
                    "line_id": crop.stem,
                    "crop_path": str(crop.resolve()),
                    "effocr_text": effocr_text,
                })
    else:
        per_page = max(1, n // len(pages))
        for page_id in pages:
            crops = by_page[page_id]
            indices = []
            if len(crops) <= per_page:
                indices = list(range(len(crops)))
            else:
                step = len(crops) / per_page
                indices = [int(i * step) for i in range(per_page)]
                extras = rng.sample(range(len(crops)), min(3, len(crops)))
                indices = list(set(indices + extras))[:per_page + 2]

            for idx in sorted(indices):
                crop = crops[idx]
                effocr_text = labels.get(str(crop.resolve()), "")
                selected.append({
                    "page_id": page_id,
                    "line_id": crop.stem,
                    "crop_path": str(crop.resolve()),
                    "effocr_text": effocr_text,
                })

    # Trim to requested size, or return all if n == 0
    if n > 0 and len(selected) > n:
        selected = selected[:n]

    return selected


async def verify_line(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    line: dict,
) -> dict:
    async with semaphore:
        img_b64 = image_to_base64(Path(line["crop_path"]))

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        for attempt in range(3):
            try:
                resp = await client.post(
                    API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=60.0,
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]

                # Parse JSON from response
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                # Handle thinking tags
                if "<think>" in content:
                    content = content.split("</think>")[-1].strip()
                verification = json.loads(content)

                return {
                    **line,
                    "llm_transcription": verification.get("transcription", ""),
                    "llm_confidence": verification.get("confidence", 0.0),
                    "llm_flag": verification.get("flag", "unknown"),
                    "llm_note": verification.get("note", ""),
                }

            except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as e:
                if attempt == 2:
                    print(f"  FAILED {line['line_id']}: {e}")
                    return {
                        **line,
                        "llm_transcription": "",
                        "llm_confidence": 0.0,
                        "llm_flag": "error",
                        "llm_note": str(e),
                    }
                await asyncio.sleep(2 ** attempt)

    return {**line, "llm_flag": "error"}


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1,
                        prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1))
    return dp[n] / m


async def main():
    n = 50
    if "--all" in sys.argv:
        n = 0  # process everything
    elif len(sys.argv) > 1 and sys.argv[1] == "--n":
        n = int(sys.argv[2])

    print(f"Selecting {n} diverse line crops...")
    sample = select_sample(n)
    print(f"Selected {len(sample)} lines across {len(set(s['page_id'] for s in sample))} pages")

    # Load already-processed lines for resume support
    already_done = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                key = f"{entry['page_id']}__{entry['line_id']}"
                already_done.add(key)

    remaining = [s for s in sample if f"{s['page_id']}__{s['line_id']}" not in already_done]
    print(f"\nSending to {MODEL} via OpenRouter...")
    print(f"Already done: {len(already_done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All lines already processed.")
        results = []
        for line in OUTPUT_PATH.read_text().splitlines():
            if line.strip():
                results.append(json.loads(line))
    else:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        results = []
        # Load existing results
        if OUTPUT_PATH.exists():
            for line in OUTPUT_PATH.read_text().splitlines():
                if line.strip():
                    results.append(json.loads(line))

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient() as client:
            tasks = [verify_line(client, semaphore, line) for line in remaining]
            with open(OUTPUT_PATH, "a") as f:
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro
                    results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    flag = result["llm_flag"]
                    conf = result["llm_confidence"]
                    print(f"  [{len(already_done)+i+1}/{len(sample)}] {result['line_id']} — {flag} (conf={conf:.2f})")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("BATCH VERIFICATION RESULTS")
    print("=" * 70)

    # Flag distribution
    flags = {}
    for r in results:
        flags[r["llm_flag"]] = flags.get(r["llm_flag"], 0) + 1
    print(f"\nFlag distribution ({len(results)} lines):")
    for flag, count in sorted(flags.items(), key=lambda x: -x[1]):
        print(f"  {flag}: {count} ({count/len(results)*100:.0f}%)")

    # Confidence distribution
    confs = [r["llm_confidence"] for r in results if r["llm_flag"] != "error"]
    if confs:
        print(f"\nConfidence: mean={sum(confs)/len(confs):.2f}, "
              f"min={min(confs):.2f}, max={max(confs):.2f}")

    # CER: LLM vs EffOCR (where both have text)
    cer_values = []
    for r in results:
        if r["llm_transcription"] and r["effocr_text"] and r["llm_flag"] not in ("error", "illegible", "not_text"):
            c = cer(r["llm_transcription"], r["effocr_text"])
            cer_values.append(c)
    if cer_values:
        print(f"\nCER (LLM vs EffOCR): mean={sum(cer_values)/len(cer_values):.3f} ({len(cer_values)} lines)")

    # Show some interesting examples
    print("\n--- SAMPLE COMPARISONS ---")
    clean = [r for r in results if r["llm_flag"] == "clean" and r["effocr_text"]]
    flagged = [r for r in results if r["llm_flag"] not in ("clean", "error")]

    print("\nClean lines (LLM agrees text is good):")
    for r in clean[:5]:
        c = cer(r["llm_transcription"], r["effocr_text"])
        match = "✓" if c < 0.05 else f"CER={c:.2f}"
        print(f"  [{match}] LLM: {r['llm_transcription'][:80]}")
        if c >= 0.05:
            print(f"         EFF: {r['effocr_text'][:80]}")

    if flagged:
        print(f"\nFlagged lines ({len(flagged)} total):")
        for r in flagged[:10]:
            print(f"  [{r['llm_flag']}] {r['line_id']}: {r['llm_note'] or r['llm_transcription'][:60]}")

    print(f"\nFull results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

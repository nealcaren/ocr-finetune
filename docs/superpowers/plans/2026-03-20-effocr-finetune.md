# EfficientOCR Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune EfficientOCR's char and word recognizers to work on reduced-resolution historical newspaper scans, using LLM-verified gold-standard transcriptions from full-res LOC JP2s.

**Architecture:** Fork efficient_ocr to remove stale dependencies. Run EffOCR on full-res JP2s to get line/word/char crops + OCR text. Verify line-level OCR with Qwen3-VL via OpenRouter. Build multi-resolution training data from verified lines. Fine-tune char + word recognizers. Evaluate against baseline.

**Design correction:** The AS pipeline only outputs region-level bounding boxes (no line/word/char boxes). Instead, we run EfficientOCR directly and extract line/word/char data from `EffOCRResult.preds` (a nested dictionary indexed by `[bbox_idx][line_idx]` containing word/char crops and predictions). Note: `save_crops` is not implemented in upstream EffOCR, so we save crops ourselves from the preds data. The AS pipeline is not needed for the pilot.

**Tech Stack:** Python 3.12+, uv, efficient_ocr (forked), Pillow (with openjp2 for JP2), OpenRouter API (Qwen3-VL 235B), timm, faiss-cpu, onnxruntime

**Spec:** `docs/superpowers/specs/2026-03-20-effocr-finetune-design.md`

---

## File Structure

```
scripts/effocr/
  sample_pages.py          # Step 1: Select diverse JP2s from LOC downloads
  run_effocr_extraction.py # Step 2: Run EffOCR on full-res JP2s, save line/word/char crops from preds
  verify_lines.py          # Step 3: LLM verification of line-level OCR via OpenRouter
  build_training_data.py   # Step 4: Build multi-resolution char/word training folders
  finetune_effocr.py       # Step 5: Run EffOCR training on prepared data
  eval_effocr.py           # Step 6: Evaluate baseline vs fine-tuned at multiple resolutions
  utils.py                 # Shared helpers (JP2 loading, path conventions)

data/effocr/               # All pipeline outputs (gitignored)
  pilot_pages.json         # Selected page paths + metadata
  extractions/             # Per-page directories with line/word/char crops + metadata JSON
  verified_lines.jsonl     # LLM verification results
  training_data/           # Final training folders
    char/                  # Character crops by ASCII code
    word/                  # Word crops by word text
```

**Important API notes:**
- `EffOCR.infer()` returns a list with ONE `EffOCRResult` per input image (not per line)
- `result.text` is the full-page text (lines joined by `\n`)
- `result.preds` is a nested dict: `preds[bbox_idx][line_idx]` → dict with `'words'` and `'chars'` entries
- Each word/char entry contains `(image_crop, coordinates)` tuples
- `save_crops` parameter is documented but **not implemented** in upstream — we extract crops from `preds` ourselves
- Training config uses flat keys: `'num_epochs'`, `'lr'`, `'batch_size'` directly under `Recognizer.char`/`Recognizer.word` (not nested under `'training'`)
- All scripts use `sys.path` insertion to handle imports from `scripts/effocr/`

---

### Task 1: Fork and Modernize EfficientOCR

This task is done in a **separate repository** (the fork). The goal is to make `efficient_ocr` installable on Python 3.12+ without yolov5 or setuptools<70, and to ensure we can access all intermediate data (line/word/char crops) from inference results.

**Repo:** Fork `dell-research-harvard/efficient_ocr` to your GitHub (e.g. `NealCaren/efficient_ocr`)

**Scope of fork changes:**
1. Lazy yolov5 imports (only loaded when yolov5 backend is explicitly requested)
2. Make yolov5 an optional dependency
3. Remove setuptools<70 constraint
4. Check `pytorch_metric_learning`, `huggingface_hub`, `transformers` pins for Python 3.12 compat
5. Verify `EffOCRResult.preds` exposes word/char crops and coordinates (this is what we use instead of `save_crops`)

- [ ] **Step 1: Fork the repo on GitHub**

Go to https://github.com/dell-research-harvard/efficient_ocr and fork it.

- [ ] **Step 2: Clone the fork locally**

```bash
cd /Users/nealcaren/Documents/GitHub
git clone https://github.com/NealCaren/efficient_ocr.git
cd efficient_ocr
```

- [ ] **Step 3: Read the source to understand the import chain**

Read these files to map the exact import chain:
- `src/efficient_ocr/__init__.py`
- `src/efficient_ocr/model/effocr.py` (EffOCR class, `infer()` and `train()`)
- `src/efficient_ocr/detection/linemodel.py` (yolov5 import)
- `src/efficient_ocr/detection/localizermodel.py` (yolov5 import)
- `src/efficient_ocr/recognition/recognizermodel.py` (training code)
- `setup.py` (dependency pins)

- [ ] **Step 4: Make detection imports lazy in `__init__.py`**

The top-level `__init__.py` unconditionally does `from .detection import LineModel` which triggers `import yolov5`. Move detection imports into `EffOCR.__init__()`, guarded by backend config.

- [ ] **Step 5: Guard yolov5 imports in detection modules**

In `linemodel.py` and `localizermodel.py`, move `import yolov5` from the top-level into the methods that use it (training methods and yolov5-backend inference).

- [ ] **Step 6: Update `setup.py` dependencies**

- Move `yolov5` to `extras_require['detection']`
- Remove any `setuptools<70` pin
- Relax `huggingface_hub` and `transformers` pins (check if current versions work)
- Check that `pytorch_metric_learning==1.6.3` installs on Python 3.12; if not, find compatible version

```python
extras_require={
    'detection': ['yolov5<=7.0.10'],
},
```

- [ ] **Step 7: Test import without yolov5 installed**

```bash
uv venv --python 3.12 test_env
source test_env/bin/activate
uv pip install -e .
python -c "from efficient_ocr import EffOCR; print('Import OK')"
```

Expected: No yolov5 error, import succeeds.

- [ ] **Step 8: Test ONNX inference and inspect preds structure**

```python
from efficient_ocr import EffOCR
effocr = EffOCR(config={
    'Recognizer': {
        'char': {'model_backend': 'onnx', 'model_dir': './models',
                 'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer'},
        'word': {'model_backend': 'onnx', 'model_dir': './models',
                 'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer'},
    },
    'Localizer': {'model_backend': 'onnx', 'model_dir': './models',
                  'hf_repo_id': 'dell-research-harvard/effocr_en'},
    'Line': {'model_backend': 'onnx', 'model_dir': './models',
             'hf_repo_id': 'dell-research-harvard/effocr_en'},
})
results = effocr.infer('path/to/test_image.jpg')
result = results[0]  # One result per image
print("Full text:", result.text[:200])
print("Preds keys:", list(result.preds.keys())[:5])
# Inspect the nested structure
for bbox_idx in list(result.preds.keys())[:2]:
    for line_idx in list(result.preds[bbox_idx].keys())[:2]:
        line_data = result.preds[bbox_idx][line_idx]
        print(f"  preds[{bbox_idx}][{line_idx}] keys: {list(line_data.keys())}")
        # Document what 'words' and 'chars' contain
```

This step is critical: **document the exact preds structure** before proceeding. The extraction script (Task 4) depends on understanding what `preds[bbox_idx][line_idx]['words']` and `['chars']` contain.

- [ ] **Step 9: Test training path for char recognizer**

Verify that `effocr.train(target='char_recognizer')` works without importing yolov5 (it uses timm, not yolov5).

- [ ] **Step 10: Commit and push**

```bash
git add -A
git commit -m "Modernize: lazy yolov5 imports, optional detection dependency, Python 3.12+ compat"
git push origin main
```

---

### Task 2: Project Setup and Environment

**Files:**
- Create: `scripts/effocr/utils.py`
- Create: `data/effocr/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Verify JP2 support**

```bash
brew install openjpeg  # if not already installed
python3 -c "from PIL import Image; img = Image.open('/Volumes/Lightning/chronicling-america/loc_downloads/sn84025829/1847-11-05/seq-1.jp2'); print(img.size)"
```

Expected: Prints image dimensions (something like `(6867, 8840)`).

- [ ] **Step 2: Create the project environment**

```bash
cd /Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune
uv venv --python 3.12 effocr_env
source effocr_env/bin/activate
uv pip install git+https://github.com/NealCaren/efficient_ocr.git  # the fork
uv pip install pillow onnxruntime httpx  # httpx for OpenRouter API
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p scripts/effocr data/effocr
touch scripts/effocr/__init__.py
```

- [ ] **Step 4: Add data/effocr to .gitignore**

Add `data/effocr/` to `.gitignore` (training data is large, not committed).

- [ ] **Step 5: Write `scripts/effocr/utils.py`**

```python
"""Shared utilities for EfficientOCR fine-tuning pipeline."""

import json
from pathlib import Path
from PIL import Image

LOC_DOWNLOADS = Path("/Volumes/Lightning/chronicling-america/loc_downloads")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"

RESOLUTIONS = {
    "75pct": 0.75,
    "50pct": 0.50,
    "30pct": 0.30,
}


def load_jp2(path: str | Path) -> Image.Image:
    """Load a JP2 image, returning a PIL Image."""
    return Image.open(path)


def downscale_image(img: Image.Image, scale: float) -> Image.Image:
    """Downscale a PIL Image by a given factor using Lanczos resampling."""
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_issue_metadata(issue_dir: Path) -> dict:
    """Load issue.json from a LOC issue directory."""
    issue_json = issue_dir / "issue.json"
    if issue_json.exists():
        return json.loads(issue_json.read_text())
    return {}
```

- [ ] **Step 6: Commit**

```bash
git add scripts/effocr/utils.py scripts/effocr/__init__.py data/effocr/.gitkeep .gitignore
git commit -m "feat(effocr): add project structure and shared utilities"
```

---

### Task 3: Sample Pages

**Files:**
- Create: `scripts/effocr/sample_pages.py`

- [ ] **Step 1: Write `sample_pages.py`**

```python
"""Select a diverse set of JP2 pages from LOC downloads for the pilot.

Samples 10 pages spread across different titles and decades.
Outputs: data/effocr/pilot_pages.json
"""

import json
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import LOC_DOWNLOADS, DATA_DIR, load_issue_metadata

NUM_PAGES = 10
SEED = 42


def collect_all_pages() -> list[dict]:
    """Walk LOC downloads and collect page metadata."""
    pages = []
    for lccn_dir in sorted(LOC_DOWNLOADS.iterdir()):
        if not lccn_dir.is_dir():
            continue
        lccn = lccn_dir.name
        for issue_dir in sorted(lccn_dir.iterdir()):
            if not issue_dir.is_dir():
                continue
            # Extract date from directory name (YYYY-MM-DD format)
            date_str = issue_dir.name
            try:
                year = int(date_str.split("-")[0])
            except (ValueError, IndexError):
                continue
            # Find JP2 files
            jp2s = sorted(issue_dir.glob("seq-*.jp2"))
            for jp2 in jp2s:
                pages.append({
                    "path": str(jp2),
                    "lccn": lccn,
                    "date": date_str,
                    "year": year,
                    "decade": (year // 10) * 10,
                    "seq": jp2.stem,
                })
    return pages


def sample_diverse(pages: list[dict], n: int) -> list[dict]:
    """Sample n pages spread across titles and decades."""
    rng = random.Random(SEED)

    # Group by decade
    by_decade = defaultdict(list)
    for p in pages:
        by_decade[p["decade"]].append(p)

    # Pick from as many decades as possible, then fill
    decades = sorted(by_decade.keys())
    selected = []
    seen_lccns = set()

    # Round-robin across decades, preferring unseen titles
    while len(selected) < n and decades:
        for decade in list(decades):
            if len(selected) >= n:
                break
            candidates = [p for p in by_decade[decade] if p["lccn"] not in seen_lccns]
            if not candidates:
                candidates = by_decade[decade]
            if not candidates:
                decades.remove(decade)
                continue
            pick = rng.choice(candidates)
            selected.append(pick)
            seen_lccns.add(pick["lccn"])
            by_decade[decade].remove(pick)

    return selected


def main():
    print("Scanning LOC downloads...")
    pages = collect_all_pages()
    print(f"Found {len(pages)} JP2 pages across {len(set(p['lccn'] for p in pages))} titles")

    selected = sample_diverse(pages, NUM_PAGES)

    print(f"\nSelected {len(selected)} pages:")
    for p in selected:
        print(f"  {p['lccn']} / {p['date']} / {p['seq']} (decade: {p['decade']}s)")

    output_path = DATA_DIR / "pilot_pages.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune
source effocr_env/bin/activate
python scripts/effocr/sample_pages.py
```

Expected: Prints 10 selected pages across different titles/decades. Creates `data/effocr/pilot_pages.json`.

- [ ] **Step 3: Inspect the output**

```bash
cat data/effocr/pilot_pages.json | python -m json.tool
```

Verify: 10 pages, multiple LCCNs, spread across decades.

- [ ] **Step 4: Commit**

```bash
git add scripts/effocr/sample_pages.py
git commit -m "feat(effocr): add page sampling script for pilot"
```

---

### Task 4: Run EfficientOCR Extraction on Full-Res JP2s

**Files:**
- Create: `scripts/effocr/run_effocr_extraction.py`

This runs EfficientOCR on each pilot page at full resolution. Since `save_crops` is not implemented in upstream EffOCR, we extract line/word/char crops and metadata from `EffOCRResult.preds` ourselves.

**Key API facts:**
- `effocr.infer(img_path)` returns a list with ONE `EffOCRResult` per input image
- `result.text` = full-page text (lines joined by `\n`)
- `result.preds` = nested dict: `preds[bbox_idx][line_idx]` → dict with word/char data
- The exact structure of word/char entries in preds will be discovered in Task 1 Step 8

- [ ] **Step 1: Write `run_effocr_extraction.py`**

```python
"""Run EfficientOCR on full-res JP2s and save all intermediate data.

For each pilot page:
- Runs EffOCR line detection, localization, and recognition
- Extracts line/word/char crops from result.preds
- Saves crops to disk with a metadata JSON

Input: data/effocr/pilot_pages.json
Output: data/effocr/extractions/{page_id}/
  - full_res.jpg (converted from JP2)
  - metadata.json (per-line OCR text, word/char info)
  - lines/ (line crop images)
  - words/ (word crop images)
  - chars/ (character crop images)
"""

import json
import sys
from pathlib import Path

# Handle imports from scripts/effocr/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, load_jp2

from efficient_ocr import EffOCR


def get_effocr() -> EffOCR:
    """Initialize EffOCR with ONNX models for all components."""
    return EffOCR(config={
        'Recognizer': {
            'char': {
                'model_backend': 'onnx',
                'model_dir': str(DATA_DIR / 'models'),
                'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer',
            },
            'word': {
                'model_backend': 'onnx',
                'model_dir': str(DATA_DIR / 'models'),
                'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer',
            },
        },
        'Localizer': {
            'model_backend': 'onnx',
            'model_dir': str(DATA_DIR / 'models'),
            'hf_repo_id': 'dell-research-harvard/effocr_en',
        },
        'Line': {
            'model_backend': 'onnx',
            'model_dir': str(DATA_DIR / 'models'),
            'hf_repo_id': 'dell-research-harvard/effocr_en',
        },
    })


def page_id(page: dict) -> str:
    """Generate a unique ID for a page."""
    return f"{page['lccn']}_{page['date']}_{page['seq']}"


def extract_crops_from_preds(result, output_dir: Path) -> list[dict]:
    """Extract line/word/char crops from EffOCRResult.preds and save to disk.

    NOTE: The exact structure of result.preds was documented in Task 1 Step 8.
    This code must be updated to match what was discovered there.
    The general pattern is: preds[bbox_idx][line_idx] -> dict with 'words'/'chars'.
    """
    lines_dir = output_dir / "lines"
    words_dir = output_dir / "words"
    chars_dir = output_dir / "chars"
    lines_dir.mkdir(exist_ok=True)
    words_dir.mkdir(exist_ok=True)
    chars_dir.mkdir(exist_ok=True)

    lines_metadata = []
    line_counter = 0

    # Split full text into lines for per-line text
    line_texts = result.text.split("\n") if result.text else []

    for bbox_idx in sorted(result.preds.keys()):
        bbox_data = result.preds[bbox_idx]
        for line_idx in sorted(bbox_data.keys()):
            line_data = bbox_data[line_idx]
            line_id = f"line_{line_counter:04d}"

            # Save line crop if available
            # ADJUST: exact key names depend on Task 1 Step 8 findings
            line_crop = line_data.get('line_image') or line_data.get('image')
            if line_crop is not None:
                line_path = lines_dir / f"{line_id}.png"
                # line_crop may be a PIL Image or numpy array
                if hasattr(line_crop, 'save'):
                    line_crop.save(str(line_path))
                else:
                    from PIL import Image
                    Image.fromarray(line_crop).save(str(line_path))

            # Extract word crops
            word_entries = []
            words = line_data.get('words', [])
            for word_idx, word_item in enumerate(words):
                # ADJUST: word_item structure depends on Task 1 Step 8
                # Expected: tuple of (image_crop, text, coords) or similar
                word_id = f"{line_id}_w{word_idx:03d}"
                if isinstance(word_item, (tuple, list)) and len(word_item) >= 2:
                    word_img, word_text = word_item[0], word_item[1]
                    word_path = words_dir / f"{word_id}.png"
                    if hasattr(word_img, 'save'):
                        word_img.save(str(word_path))
                    elif word_img is not None:
                        from PIL import Image
                        Image.fromarray(word_img).save(str(word_path))
                    word_entries.append({
                        "word_id": word_id,
                        "text": str(word_text),
                        "path": str(word_path),
                    })

            # Extract char crops
            char_entries = []
            chars = line_data.get('chars', [])
            for char_idx, char_item in enumerate(chars):
                char_id = f"{line_id}_c{char_idx:03d}"
                if isinstance(char_item, (tuple, list)) and len(char_item) >= 2:
                    char_img, char_label = char_item[0], char_item[1]
                    char_path = chars_dir / f"{char_id}.png"
                    if hasattr(char_img, 'save'):
                        char_img.save(str(char_path))
                    elif char_img is not None:
                        from PIL import Image
                        Image.fromarray(char_img).save(str(char_path))
                    char_entries.append({
                        "char_id": char_id,
                        "label": str(char_label),
                        "path": str(char_path),
                    })

            # Get line-level text
            line_text = line_texts[line_counter] if line_counter < len(line_texts) else ""

            lines_metadata.append({
                "line_id": line_id,
                "line_index": line_counter,
                "bbox_idx": bbox_idx,
                "line_idx_in_bbox": line_idx,
                "text": line_text,
                "words": word_entries,
                "chars": char_entries,
                "line_crop_path": str(lines_dir / f"{line_id}.png"),
            })

            line_counter += 1

    return lines_metadata


def main():
    pilot_path = DATA_DIR / "pilot_pages.json"
    if not pilot_path.exists():
        print("Run sample_pages.py first")
        sys.exit(1)

    pages = json.loads(pilot_path.read_text())
    extractions_dir = DATA_DIR / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EfficientOCR models...")
    effocr = get_effocr()

    for i, page in enumerate(pages):
        pid = page_id(page)
        output_dir = extractions_dir / pid
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            print(f"[{i+1}/{len(pages)}] {pid} — already extracted, skipping")
            continue

        print(f"[{i+1}/{len(pages)}] Processing {pid}...")

        # Load the full-res JP2 and convert to JPG (EffOCR may not handle JP2)
        jp2_path = page["path"]
        img = load_jp2(jp2_path)
        jpg_path = output_dir / "full_res.jpg"
        img.save(str(jpg_path), "JPEG", quality=95)

        # Run EffOCR — returns one result per image
        results = effocr.infer(str(jpg_path))
        result = results[0]

        # Extract all crops from preds
        lines_metadata = extract_crops_from_preds(result, output_dir)

        # Save metadata
        metadata = {
            "page_id": pid,
            "jp2_path": page["path"],
            "full_text": result.text,
            "num_lines": len(lines_metadata),
            "lines": lines_metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        print(f"  → {len(lines_metadata)} lines extracted, saved to {output_dir}")

    print(f"\nDone. Extractions in {extractions_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on ONE pilot page first**

```bash
cd /Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune
source effocr_env/bin/activate
python scripts/effocr/run_effocr_extraction.py
```

Expected: First run downloads ONNX models (~200MB). Processes pages, saving crops and metadata.

- [ ] **Step 3: STOP — Inspect output and fix extraction code**

```bash
ls data/effocr/extractions/
ls data/effocr/extractions/$(ls data/effocr/extractions/ | head -1)/
cat data/effocr/extractions/$(ls data/effocr/extractions/ | head -1)/metadata.json | python -m json.tool | head -80
ls data/effocr/extractions/$(ls data/effocr/extractions/ | head -1)/lines/ | head -10
ls data/effocr/extractions/$(ls data/effocr/extractions/ | head -1)/words/ | head -10
ls data/effocr/extractions/$(ls data/effocr/extractions/ | head -1)/chars/ | head -10
```

**This is the most important checkpoint.** The `extract_crops_from_preds()` function uses placeholder access patterns (marked with `# ADJUST` comments). After inspecting the actual output:
1. Verify line/word/char crops were saved correctly
2. Verify the metadata JSON has accurate text and paths
3. Fix any mismatches in how preds are accessed

Do NOT proceed to Task 5 until the extraction output is correct.

- [ ] **Step 4: Commit**

```bash
git add scripts/effocr/run_effocr_extraction.py
git commit -m "feat(effocr): add full-res EffOCR extraction script"
```

---

### Task 5: LLM Line Verification

**Files:**
- Create: `scripts/effocr/verify_lines.py`

- [ ] **Step 1: Write `verify_lines.py`**

```python
"""Verify line-level OCR quality using Qwen3-VL 235B via OpenRouter.

For each line crop from EffOCR extraction:
- Sends the line image to LLM for blind transcription
- Compares LLM transcription against EffOCR output
- Classifies as accept / correct / reject

Input: data/effocr/extractions/*/crops/ and lines.json
Output: data/effocr/verified_lines.jsonl

Resume-safe: skips already-processed lines on restart.
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT = 5
OUTPUT_PATH = DATA_DIR / "verified_lines.jsonl"

VERIFY_PROMPT = """Look at this image of a single line of text from a historical newspaper.

Step 1: Transcribe exactly what you see in the image. Include all punctuation and capitalization.

Step 2: Compare your transcription with this OCR output: "{ocr_text}"

Step 3: Respond with ONLY a JSON object (no markdown, no explanation):
{{"status": "accept", "transcription": "your transcription", "confidence": 0.95}}

Where:
- status is "accept" if the OCR output matches your transcription
- status is "correct" if you can read the text but the OCR is wrong (use YOUR transcription)
- status is "reject" if the image is unreadable or you are not confident
- confidence is 0.0 to 1.0 for your transcription quality"""


def image_to_base64(path: Path) -> str:
    """Read image file and return base64 data URL."""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def load_processed_ids(output_path: Path) -> set[str]:
    """Load already-processed line IDs from JSONL output."""
    processed = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                processed.add(entry["line_id"])
    return processed


def collect_lines() -> list[dict]:
    """Collect all line crops and their EffOCR text from extraction metadata."""
    lines = []
    extractions_dir = DATA_DIR / "extractions"
    if not extractions_dir.exists():
        print("Run run_effocr_extraction.py first")
        sys.exit(1)

    for page_dir in sorted(extractions_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        metadata_path = page_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        page_id = metadata["page_id"]

        for line_info in metadata.get("lines", []):
            crop_path = line_info.get("line_crop_path", "")
            if not crop_path or not Path(crop_path).exists():
                continue
            line_id = f"{page_id}__{line_info['line_id']}"
            lines.append({
                "line_id": line_id,
                "crop_path": crop_path,
                "ocr_text": line_info.get("text", ""),
                "page_dir": str(page_dir),
            })

    return lines


async def verify_line(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    line: dict,
) -> dict:
    """Send a single line to OpenRouter for verification."""
    async with semaphore:
        img_b64 = image_to_base64(Path(line["crop_path"]))
        prompt = VERIFY_PROMPT.format(ocr_text=line["ocr_text"])

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 300,
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

                # Parse JSON from response (handle markdown wrapping)
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                verification = json.loads(content)

                return {
                    "line_id": line["line_id"],
                    "crop_path": line["crop_path"],
                    "ocr_text": line["ocr_text"],
                    "status": verification.get("status", "reject"),
                    "transcription": verification.get("transcription", ""),
                    "confidence": verification.get("confidence", 0.0),
                }

            except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as e:
                if attempt == 2:
                    print(f"  Failed after 3 attempts for {line['line_id']}: {e}")
                    return {
                        "line_id": line["line_id"],
                        "crop_path": line["crop_path"],
                        "ocr_text": line["ocr_text"],
                        "status": "reject",
                        "transcription": "",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                await asyncio.sleep(2 ** attempt)

    # Should not reach here
    return {"line_id": line["line_id"], "status": "reject", "confidence": 0.0}


async def main():
    if not OPENROUTER_API_KEY:
        print("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    lines = collect_lines()
    processed = load_processed_ids(OUTPUT_PATH)
    remaining = [l for l in lines if l["line_id"] not in processed]

    print(f"Total lines: {len(lines)}, already processed: {len(processed)}, remaining: {len(remaining)}")

    if not remaining:
        print("All lines already verified.")
        report_stats()
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        tasks = [verify_line(client, semaphore, line) for line in remaining]

        with open(OUTPUT_PATH, "a") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                f.write(json.dumps(result) + "\n")
                f.flush()
                status = result["status"]
                line_id = result["line_id"]
                print(f"  [{status}] {line_id}")

    report_stats()


def report_stats():
    """Print verification statistics."""
    if not OUTPUT_PATH.exists():
        return
    statuses = {"accept": 0, "correct": 0, "reject": 0}
    total = 0
    for line in OUTPUT_PATH.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            statuses[entry.get("status", "reject")] = statuses.get(entry.get("status", "reject"), 0) + 1
            total += 1

    print(f"\nVerification Statistics:")
    print(f"  Total: {total}")
    for status, count in sorted(statuses.items()):
        pct = (count / total * 100) if total else 0
        print(f"  {status}: {count} ({pct:.1f}%)")
    gold = statuses.get("accept", 0) + statuses.get("correct", 0)
    print(f"  Gold-standard lines: {gold} ({gold/total*100:.1f}%)" if total else "")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Test with a small batch**

```bash
export OPENROUTER_API_KEY="your-key-here"
python scripts/effocr/verify_lines.py
```

Expected: Processes lines with accept/correct/reject outcomes. Prints statistics at the end. Results saved incrementally to `data/effocr/verified_lines.jsonl`.

- [ ] **Step 3: Inspect results and statistics**

```bash
head -5 data/effocr/verified_lines.jsonl | python -m json.tool
```

Check: acceptance rate is reasonable (>50%), rejections make sense (truly illegible lines), corrections look plausible.

- [ ] **Step 4: Commit**

```bash
git add scripts/effocr/verify_lines.py
git commit -m "feat(effocr): add LLM line verification via OpenRouter"
```

---

### Task 6: Build Multi-Resolution Training Data

**Files:**
- Create: `scripts/effocr/build_training_data.py`

**Important:** This task depends on understanding the exact output format from Task 4 (the COCO annotations and crop directory structure). The code below uses placeholder patterns for crop file naming — these must be adjusted after inspecting the actual EffOCR output.

- [ ] **Step 1: Write `build_training_data.py`**

```python
"""Build EfficientOCR training data from verified lines.

Uses extraction metadata (not COCO annotations, which lack the needed detail).
For each verified line, loads the original full-res JP2, downscales to
multiple resolutions, and re-crops characters and words at each scale.

Excludes test pages (last 2) from training data.

Input:
  data/effocr/verified_lines.jsonl
  data/effocr/extractions/*/metadata.json
  data/effocr/pilot_pages.json
Output:
  data/effocr/training_data/char/{ascii_code}/PAIRED_*.png
  data/effocr/training_data/word/{word_text}/PAIRED_*.png
"""

import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESOLUTIONS, load_jp2, downscale_image

TEST_PAGE_COUNT = 2  # Last N pages are held out for evaluation


def load_verified_lines() -> dict[str, dict]:
    """Load verified lines, keyed by line_id. Only accept/correct."""
    verified = {}
    vl_path = DATA_DIR / "verified_lines.jsonl"
    if not vl_path.exists():
        print("Run verify_lines.py first")
        sys.exit(1)
    for line in vl_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") in ("accept", "correct"):
            verified[entry["line_id"]] = entry
    return verified


def get_test_page_ids() -> set[str]:
    """Get page IDs for test pages (excluded from training)."""
    pilot = json.loads((DATA_DIR / "pilot_pages.json").read_text())
    test_pages = pilot[-TEST_PAGE_COUNT:]
    return {f"{p['lccn']}_{p['date']}_{p['seq']}" for p in test_pages}


def safe_folder_name(text: str) -> str:
    """Sanitize text for use as a folder name."""
    text = text.strip().lower()
    return re.sub(r'[^a-z0-9_-]', '_', text)


def process_page(page_dir: Path, verified: dict, output_dir: Path,
                 pilot_pages: list[dict]):
    """Process one page using extraction metadata and re-cropping from JP2.

    For each verified line's chars/words, we:
    1. Load the full-res JP2
    2. Downscale to each target resolution
    3. Load the original full-res char/word crop (saved by extraction)
    4. Save scaled versions in the EfficientOCR training folder structure
    """
    metadata_path = page_dir / "metadata.json"
    if not metadata_path.exists():
        return 0, 0

    metadata = json.loads(metadata_path.read_text())
    pid = metadata["page_id"]

    char_count = 0
    word_count = 0

    for line_info in metadata.get("lines", []):
        line_id = f"{pid}__{line_info['line_id']}"

        # Only use verified lines
        if line_id not in verified:
            continue

        verified_entry = verified[line_id]
        # Use LLM transcription for "correct" status, EffOCR text for "accept"
        gold_text = verified_entry["transcription"]

        # Process character crops
        for char_info in line_info.get("chars", []):
            char_label = char_info.get("label", "")
            if not char_label or len(char_label) != 1:
                continue
            char_path = Path(char_info["path"])
            if not char_path.exists():
                continue

            full_crop = Image.open(str(char_path))
            ascii_code = ord(char_label)
            char_dir = output_dir / "char" / str(ascii_code)
            char_dir.mkdir(parents=True, exist_ok=True)

            # Save at each resolution (downscale the crop itself)
            for res_name, scale in RESOLUTIONS.items():
                scaled = full_crop.resize(
                    (max(1, int(full_crop.width * scale)),
                     max(1, int(full_crop.height * scale))),
                    Image.LANCZOS
                )
                filename = f"PAIRED_{pid}_{char_info['char_id']}_{res_name}.png"
                scaled.save(str(char_dir / filename))
                char_count += 1

        # Process word crops
        for word_info in line_info.get("words", []):
            word_text = word_info.get("text", "")
            if not word_text or len(word_text) > 50:
                continue
            word_path = Path(word_info["path"])
            if not word_path.exists():
                continue

            full_crop = Image.open(str(word_path))
            safe_word = safe_folder_name(word_text)
            if not safe_word:
                continue
            word_dir = output_dir / "word" / safe_word
            word_dir.mkdir(parents=True, exist_ok=True)

            for res_name, scale in RESOLUTIONS.items():
                scaled = full_crop.resize(
                    (max(1, int(full_crop.width * scale)),
                     max(1, int(full_crop.height * scale))),
                    Image.LANCZOS
                )
                filename = f"PAIRED_{pid}_{word_info['word_id']}_{res_name}.png"
                scaled.save(str(word_dir / filename))
                word_count += 1

    return char_count, word_count


def main():
    verified = load_verified_lines()
    print(f"Loaded {len(verified)} verified gold-standard lines")

    test_ids = get_test_page_ids()
    print(f"Excluding test pages: {test_ids}")

    pilot_pages = json.loads((DATA_DIR / "pilot_pages.json").read_text())
    output_dir = DATA_DIR / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_words = 0

    extractions_dir = DATA_DIR / "extractions"
    for page_dir in sorted(extractions_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        # Skip test pages
        if page_dir.name in test_ids:
            print(f"  {page_dir.name}: SKIPPED (test page)")
            continue
        chars, words = process_page(page_dir, verified, output_dir, pilot_pages)
        total_chars += chars
        total_words += words
        print(f"  {page_dir.name}: {chars} char crops, {words} word crops")

    char_classes = len(list((output_dir / "char").iterdir())) if (output_dir / "char").exists() else 0
    word_classes = len(list((output_dir / "word").iterdir())) if (output_dir / "word").exists() else 0

    print(f"\nTraining data summary:")
    print(f"  Character crops: {total_chars} across {char_classes} classes")
    print(f"  Word crops: {total_words} across {word_classes} classes")
    print(f"  Resolutions: {', '.join(RESOLUTIONS.keys())}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python scripts/effocr/build_training_data.py
```

Expected: Creates `data/effocr/training_data/char/` and `data/effocr/training_data/word/` with crops at 3 resolutions.

- [ ] **Step 3: Inspect training data structure**

```bash
# Check character class distribution
ls data/effocr/training_data/char/ | wc -l
for d in data/effocr/training_data/char/*/; do echo "$(basename $d) ($(python3 -c "print(chr($(basename $d)))")): $(ls $d | wc -l) crops"; done | head -20

# Check word class distribution
ls data/effocr/training_data/word/ | wc -l
ls data/effocr/training_data/word/ | head -20
```

- [ ] **Step 4: Commit**

```bash
git add scripts/effocr/build_training_data.py
git commit -m "feat(effocr): build multi-resolution char/word training data from verified lines"
```

---

### Task 7: Baseline Evaluation

**Files:**
- Create: `scripts/effocr/eval_effocr.py`

Run unmodified EfficientOCR on the 2 held-out test pages at each resolution to establish the baseline CER/WER before fine-tuning.

- [ ] **Step 1: Write `eval_effocr.py`**

```python
"""Evaluate EfficientOCR at multiple resolutions.

Runs EffOCR on test pages at 30%, 50%, 75%, and 100% resolution.
Computes CER and WER against gold-standard transcriptions.
Supports comparing baseline vs fine-tuned models.

Input:
  data/effocr/pilot_pages.json (last 2 pages used as test set)
  data/effocr/verified_lines.jsonl (gold-standard transcriptions)
Output:
  data/effocr/eval_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESOLUTIONS, load_jp2, downscale_image

from efficient_ocr import EffOCR

# Train/test split: last 2 of 10 pages are test
TEST_PAGE_COUNT = 2


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate using edit distance."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    # Simple Levenshtein distance
    m, n = len(reference), len(hypothesis)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1),
            )
    return dp[n] / m


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    m, n = len(ref_words), len(hyp_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + (0 if ref_words[i - 1] == hyp_words[j - 1] else 1),
            )
    return dp[n] / m


def get_test_pages() -> list[dict]:
    """Get the test pages (last 2 from pilot)."""
    pilot = json.loads((DATA_DIR / "pilot_pages.json").read_text())
    return pilot[-TEST_PAGE_COUNT:]


def get_gold_standard(page_id: str) -> dict[int, str]:
    """Get gold-standard transcriptions for a page's lines."""
    gold = {}
    vl_path = DATA_DIR / "verified_lines.jsonl"
    for line in vl_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["line_id"].startswith(page_id) and entry["status"] in ("accept", "correct"):
            # Extract line index from line_id
            parts = entry["line_id"].split("__line_")
            if len(parts) == 2:
                gold[int(parts[1])] = entry["transcription"]
    return gold


def evaluate_model(effocr: EffOCR, test_pages: list[dict], model_name: str) -> list[dict]:
    """Run evaluation at all resolutions using page-level CER/WER.

    Compares full-page OCR text against concatenated gold-standard transcriptions.
    Page-level comparison avoids fragile line-by-line index matching (line detection
    may produce different line counts at different resolutions).
    """
    results = []
    all_resolutions = {"100pct": 1.0, **RESOLUTIONS}

    for page in test_pages:
        pid = f"{page['lccn']}_{page['date']}_{page['seq']}"
        gold = get_gold_standard(pid)
        if not gold:
            print(f"  No gold standard for {pid}, skipping")
            continue

        # Concatenate gold-standard lines into page-level reference
        gold_page_text = "\n".join(gold[k] for k in sorted(gold.keys()))

        full_img = load_jp2(page["path"])

        for res_name, scale in all_resolutions.items():
            img = downscale_image(full_img, scale) if scale < 1.0 else full_img

            tmp_path = DATA_DIR / f"tmp_eval_{res_name}.jpg"
            img.save(str(tmp_path), "JPEG", quality=95)

            # infer() returns one result per image
            ocr_results = effocr.infer(str(tmp_path))
            ocr_text = ocr_results[0].text if ocr_results else ""
            tmp_path.unlink()

            c = cer(gold_page_text, ocr_text)
            w = wer(gold_page_text, ocr_text)

            results.append({
                "model": model_name,
                "page_id": pid,
                "resolution": res_name,
                "scale": scale,
                "cer": round(c, 4),
                "wer": round(w, 4),
                "gold_lines": len(gold),
            })

            print(f"  {model_name} @ {res_name} on {pid}: CER={c:.3f} WER={w:.3f}")

    return results


def main():
    model_dir = None
    model_name = "baseline"

    # Check for command-line arg pointing to fine-tuned model
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        model_name = "finetuned"
        print(f"Evaluating fine-tuned model from {model_dir}")
    else:
        print("Evaluating baseline EfficientOCR")

    test_pages = get_test_pages()
    print(f"Test pages: {len(test_pages)}")

    # Initialize EffOCR — use custom model dir if provided
    config = {
        'Recognizer': {
            'char': {
                'model_backend': 'onnx',
                'model_dir': model_dir or str(DATA_DIR / 'models'),
                'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer',
            },
            'word': {
                'model_backend': 'onnx',
                'model_dir': model_dir or str(DATA_DIR / 'models'),
                'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer',
            },
        },
        'Localizer': {
            'model_backend': 'onnx',
            'model_dir': str(DATA_DIR / 'models'),
            'hf_repo_id': 'dell-research-harvard/effocr_en',
        },
        'Line': {
            'model_backend': 'onnx',
            'model_dir': str(DATA_DIR / 'models'),
            'hf_repo_id': 'dell-research-harvard/effocr_en',
        },
    }
    effocr = EffOCR(config=config)

    results = evaluate_model(effocr, test_pages, model_name)

    # Save results
    output_path = DATA_DIR / "eval_results.json"
    existing = json.loads(output_path.read_text()) if output_path.exists() else []
    existing.extend(results)
    output_path.write_text(json.dumps(existing, indent=2))

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run baseline evaluation**

```bash
python scripts/effocr/eval_effocr.py
```

Expected: CER/WER numbers at each resolution. Full-res should be ~5% CER, 30% resolution might be ~36% CER (matching known degradation).

- [ ] **Step 3: Commit**

```bash
git add scripts/effocr/eval_effocr.py
git commit -m "feat(effocr): add multi-resolution evaluation script with CER/WER"
```

---

### Task 8: Fine-Tune EfficientOCR

**Files:**
- Create: `scripts/effocr/finetune_effocr.py`

- [ ] **Step 1: Write `finetune_effocr.py`**

```python
"""Fine-tune EfficientOCR char and word recognizers on prepared training data.

Uses EfficientOCR's built-in training framework with timm backend.

Input: data/effocr/training_data/char/ and word/
Output: data/effocr/finetuned_models/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR

from efficient_ocr import EffOCR

TRAINING_DATA = DATA_DIR / "training_data"
OUTPUT_DIR = DATA_DIR / "finetuned_models"


def train_char_recognizer():
    """Fine-tune the character recognizer."""
    char_data_dir = TRAINING_DATA / "char"
    if not char_data_dir.exists():
        print("No character training data found. Run build_training_data.py first.")
        return

    num_classes = len(list(char_data_dir.iterdir()))
    total_crops = sum(len(list(d.iterdir())) for d in char_data_dir.iterdir() if d.is_dir())
    print(f"Character training data: {total_crops} crops across {num_classes} classes")

    char_output = OUTPUT_DIR / "char_recognizer"
    char_output.mkdir(parents=True, exist_ok=True)

    # Detect available device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'cpu'  # MPS may not be supported by EffOCR training
    else:
        device = 'cpu'

    config = {
        'Global': {
            'single_model_training': 'char_recognizer',
        },
        'Recognizer': {
            'char': {
                'model_backend': 'timm',
                'model_dir': str(char_output),
                'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer',
                'batch_size': 128,
                'num_epochs': 10,
                'lr': 0.002,
                'device': device,
                'ready_to_go_data_dir_path': str(char_data_dir),
            },
        },
    }

    print("Training character recognizer...")
    effocr = EffOCR(config=config)
    effocr.train(target='char_recognizer')
    print(f"Character recognizer saved to {char_output}")


def train_word_recognizer():
    """Fine-tune the word recognizer."""
    word_data_dir = TRAINING_DATA / "word"
    if not word_data_dir.exists():
        print("No word training data found. Run build_training_data.py first.")
        return

    num_classes = len(list(word_data_dir.iterdir()))
    total_crops = sum(len(list(d.iterdir())) for d in word_data_dir.iterdir() if d.is_dir())
    print(f"Word training data: {total_crops} crops across {num_classes} classes")

    word_output = OUTPUT_DIR / "word_recognizer"
    word_output.mkdir(parents=True, exist_ok=True)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'Global': {
            'single_model_training': 'word_recognizer',
        },
        'Recognizer': {
            'word': {
                'model_backend': 'timm',
                'model_dir': str(word_output),
                'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer',
                'batch_size': 64,
                'num_epochs': 10,
                'lr': 0.002,
                'device': device,
                'ready_to_go_data_dir_path': str(word_data_dir),
            },
        },
    }

    print("Training word recognizer...")
    effocr = EffOCR(config=config)
    effocr.train(target='word_recognizer')
    print(f"Word recognizer saved to {word_output}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_char_recognizer()
    print()
    train_word_recognizer()

    print(f"\nFine-tuned models saved to {OUTPUT_DIR}")
    print("Run eval_effocr.py with the model path to evaluate:")
    print(f"  python scripts/effocr/eval_effocr.py {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run fine-tuning**

```bash
python scripts/effocr/finetune_effocr.py
```

Expected: Trains char recognizer then word recognizer. Outputs models to `data/effocr/finetuned_models/`.

Note: This requires a GPU. If running on CPU-only Mac, training will be very slow. Consider running on a machine with GPU access or adjusting `'device': 'cpu'`.

- [ ] **Step 3: Commit**

```bash
git add scripts/effocr/finetune_effocr.py
git commit -m "feat(effocr): add recognizer fine-tuning script"
```

---

### Task 9: Evaluate Fine-Tuned Model

- [ ] **Step 1: Run evaluation with fine-tuned model**

```bash
python scripts/effocr/eval_effocr.py data/effocr/finetuned_models
```

Expected: CER/WER numbers at each resolution for the fine-tuned model. Compare against baseline numbers from Task 7.

- [ ] **Step 2: Compare baseline vs fine-tuned**

```bash
python -c "
import json
results = json.load(open('data/effocr/eval_results.json'))
print(f\"{'Model':<15} {'Resolution':<12} {'CER':>8} {'WER':>8}\")
print('-' * 45)
for r in sorted(results, key=lambda x: (x['model'], x['scale'])):
    print(f\"{r['model']:<15} {r['resolution']:<12} {r['avg_cer']:>8.3f} {r['avg_wer']:>8.3f}\")
"
```

Expected: Fine-tuned model shows lower CER/WER at reduced resolutions compared to baseline.

- [ ] **Step 3: Commit evaluation results**

```bash
git add data/effocr/eval_results.json
git commit -m "results(effocr): baseline vs fine-tuned evaluation on pilot pages"
```

---

## Checkpoint Notes

**After Task 1 (fork):** Verify the fork installs and runs on Python 3.12+. This unblocks everything else.

**After Task 4 (extraction):** STOP and inspect the EffOCR output format. The exact COCO annotation schema and crop file naming will determine how Tasks 5 and 6 work. The code in those tasks uses best-guess patterns that will likely need adjustment.

**After Task 5 (verification):** Check acceptance/rejection rates. If rejection rate is very high (>50%), the EffOCR baseline OCR may be too poor on these images, and the verification prompt may need tuning.

**After Task 7 (baseline eval):** This establishes the "before" numbers. If CER at full res is already bad (~20%+), there may be issues with EffOCR setup rather than resolution degradation.

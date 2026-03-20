# EfficientOCR Fine-Tuning from LOC JP2s with LLM Verification

## Problem

EfficientOCR was trained on high-resolution JP2 scans (~6,800x8,800 pixels). At reduced resolution (~2,200x2,800), its character/word recognition degrades significantly (CER ~36% vs ~5% at full res) and its legibility classifier rejects nearly half the page. A fine-tuned version could provide fast CPU-based OCR for bulk processing of lower-resolution images.

## Approach

Use 285K locally stored full-resolution JP2 scans from LOC's Chronicling America collection as source material. Run the American Stories pipeline to extract line-level OCR, verify quality with a powerful vision LLM, then use verified lines to fine-tune EfficientOCR's recognizers at multiple resolutions.

## Data Source

- **Location**: `/Volumes/Lightning/chronicling-america/loc_downloads/`
- **Structure**: `{lccn}/{date}/seq-{N}.jp2` with `issue.json` metadata
- **Scale**: 344 titles, ~285,000 JP2 scans
- **Pilot**: 10 pages sampled across diverse titles/decades/print quality

## Pipeline

```
Full-res JP2 scans (local, from LOC)
    │
    ▼
American Stories pipeline (layout → line detection → OCR)
    │
    ▼
Line crops + bounding boxes + OCR text
    │
    ▼
Qwen3-VL 235B via OpenRouter (line-level verification)
    │  accept / reject / correct
    ▼
Gold-standard dataset (line image + verified transcription)
    │
    ▼
Extract char/word crops at multiple resolutions (~30%, ~50%, ~75%)
    │
    ▼
EfficientOCR fine-tuning (char + word recognizers)
    │
    ▼
Evaluation (CER/WER at each resolution)
```

## Step 1: Sample Pages

Select 10 pages from LOC downloads for the pilot. Criteria:
- Spread across different titles (LCCNs)
- Spread across decades (1840s–1960s)
- Mix of print quality (clear vs degraded)

Script: `sample_pages.py`

## Step 2: Run American Stories Pipeline

Run the full AS pipeline on selected full-res JP2s locally:
- Layout detection (YOLO) → article regions
- Line detection (YOLO) → text lines within regions
- Character/word localization → bounding boxes within lines
- EfficientOCR recognition → OCR text per line

This produces line crops, character-level bounding boxes, and initial OCR text.

Script: `run_as_pipeline.py` (wrapper around AS pipeline with our paths/config)

## Step 3: LLM Quality Verification

For each detected line:
1. Crop the line image from the full-res JP2
2. Send to Qwen3-VL 235B via OpenRouter with the AS OCR text
3. Ask the model to: accept the transcription as correct, reject it as unreadable/wrong, or provide a corrected transcription
4. Only accepted and corrected lines enter the gold-standard dataset

Output: JSON mapping line crop paths to verified transcriptions, with accept/reject/correct status.

Script: `verify_lines.py`

**OpenRouter config**: Qwen3-VL 235B (`qwen/qwen3-vl-235b-a22b-instruct`)

## Step 4: Build Training Data

From verified lines, construct EfficientOCR training data:

**Character recognizer data**:
- Use AS localizer bounding boxes to crop individual characters from line images
- Organize into folders by ASCII code (e.g., `65/` for 'A')
- Prefix real crops with `PAIRED_` (EfficientOCR convention)
- Repeat at 3 resolutions: ~30%, ~50%, ~75% of original

**Word recognizer data**:
- Crop words using AS word-level bounding boxes
- Pair with verified text labels
- Same multi-resolution treatment

Script: `build_effocr_training_data.py`

## Step 5: Fine-Tune EfficientOCR

Fine-tune using EfficientOCR's built-in training framework:

| Component | Action | Rationale |
|-----------|--------|-----------|
| Char recognizer | Fine-tune | Primary error source at low res |
| Word recognizer | Fine-tune | Captures word-level context |
| Localizer | Skip (revisit if needed) | Likely works at reduced res |
| Line detector | Skip | Works proportionally at reduced res |
| Layout detector | Skip | Same regions detected at both resolutions; not trainable in-package |
| Legibility classifier | Skip | Not trainable in-package |

Script: `finetune_effocr.py`

## Step 6: Evaluate

Compare original vs fine-tuned EfficientOCR on held-out pages:
- Test at each target resolution (~30%, ~50%, ~75%)
- Metrics: CER, WER
- Benchmark against GLM-OCR on same pages for reference

Script: `eval_effocr.py`

## Dependency Modernization

EfficientOCR currently requires Python 3.11 and `setuptools<70` due to yolov5's `pkg_resources` import. Attempt to:
- Patch or fork to replace `pkg_resources` usage with `importlib.metadata`
- Or swap yolov5 dependency for ultralytics (YOLOv8) which doesn't have this issue
- Goal: Python 3.12+ compatibility

## Scripts Summary

| Script | Purpose |
|--------|---------|
| `sample_pages.py` | Select diverse pages from LOC downloads |
| `run_as_pipeline.py` | Wrapper to run AS pipeline on selected JP2s |
| `verify_lines.py` | Send line crops to Qwen3-VL via OpenRouter |
| `build_effocr_training_data.py` | Extract char/word crops at multiple resolutions |
| `finetune_effocr.py` | Run EfficientOCR training |
| `eval_effocr.py` | Evaluate original vs fine-tuned |

## Infrastructure

- **All local** — no Longleaf for the pilot
- **AS pipeline**: Needs GPU for YOLO detection (local GPU or CPU with patience)
- **Qwen3-VL**: OpenRouter API (needs API key)
- **EfficientOCR training**: GPU recommended, T4-level sufficient
- **EfficientOCR inference**: CPU

## Scale-Up Plan

After the 10-page pilot validates the pipeline:
1. Strategically sample a few thousand pages across decades/quality levels
2. Run the full pipeline — a few thousand pages yields hundreds of thousands of verified lines
3. Re-evaluate and iterate on training hyperparameters

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

Script: `run_as_pipeline.py` (wrapper around AS pipeline with our paths/config)

**JP2 dependency**: Pillow's JP2 support requires `openjp2` system library. Install via `brew install openjpeg` (macOS) and verify with `from PIL import Image; Image.open("test.jp2")`. Alternatively use `glymur` for JP2 decoding.

### AS Pipeline Output Format

The AS pipeline (`run_img2txt_yolo_pipeline.py`) produces one JSON file per input image. Each JSON contains:

```json
{
  "regions": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": "article|ad|headline|...",
      "legibility_score": 0.95,
      "lines": [
        {
          "bbox": [x1, y1, x2, y2],
          "text": "OCR output for this line",
          "words": [
            {"bbox": [x1, y1, x2, y2], "text": "word", "confidence": 0.92}
          ],
          "chars": [
            {"bbox": [x1, y1, x2, y2], "char": "A", "confidence": 0.88}
          ]
        }
      ]
    }
  ]
}
```

Character-level bounding boxes come from EfficientOCR's localizer running inside the AS pipeline. These are the coordinates we use in Step 4 to crop individual characters. Line crops must be extracted from the original JP2 using line-level bounding boxes — the AS pipeline does not save crop images to disk.

## Step 3: LLM Quality Verification

For each detected line:
1. Crop the line image from the full-res JP2 using line bounding box
2. Send to Qwen3-VL 235B via OpenRouter — model transcribes the line image *first* (without seeing AS OCR), then compares to the AS text
3. Outcome: accept (AS matches LLM), correct (use LLM transcription), or reject (image unreadable / low confidence)
4. Only accepted and corrected lines enter the gold-standard dataset

**Prompt design**: The LLM transcribes blind first to avoid anchoring bias. Then it receives the AS OCR text and returns structured JSON:

```json
{"status": "accept|correct|reject", "transcription": "verified text", "confidence": 0.95}
```

For corrections, the LLM's transcription is used. Lines where the LLM itself is low-confidence are rejected.

**API handling**:
- Resume support: results saved incrementally to JSONL, script skips already-processed lines on restart
- Rate limiting: configurable concurrency (default 5 parallel requests) with exponential backoff on 429/500
- Estimated volume: ~200-400 lines per page → ~2,000-4,000 calls for 10-page pilot

**Quality statistics**: After verification, report acceptance/rejection/correction rates and common correction patterns to validate pipeline health.

Output: JSONL file mapping line crop paths to verified transcriptions with status and confidence.

Script: `verify_lines.py`

**OpenRouter config**: Qwen3-VL 235B (`qwen/qwen3-vl-235b-a22b-instruct`)

## Step 4: Build Training Data

From verified lines, construct EfficientOCR training data:

**Character recognizer data** (image classification, folder-per-class):
```
char_training_data/
  65/           # chr(65) = 'A'
    PAIRED_page1_line3_char0_30pct.png
    PAIRED_page1_line3_char0_50pct.png
    PAIRED_page1_line3_char0_75pct.png
  66/           # chr(66) = 'B'
    ...
```
- Crop characters from line images using AS localizer bounding boxes
- `PAIRED_` prefix distinguishes real crops from synthetic augmentations in EfficientOCR's training code
- Each character appears at 3 resolutions

**Word recognizer data** (same folder-per-class structure):
```
word_training_data/
  the/
    PAIRED_page1_line3_word0_30pct.png
    ...
  was/
    ...
```
- Crop words using AS word-level bounding boxes
- Folder name = verified word text

**Resolution scaling**: Downscale the *full page* first (Lanczos resampling), then crop characters/words from the downscaled page. This matches inference conditions, where the model sees a naturally low-res image rather than a downscaled crop.

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

**Baseline first**: Before fine-tuning, run unmodified EfficientOCR at each target resolution on pilot pages. Record CER/WER as the baseline.

**After fine-tuning**: Compare original vs fine-tuned EfficientOCR on held-out pages:
- Test at each target resolution (~30%, ~50%, ~75%)
- Metrics: CER, WER
- Benchmark against GLM-OCR on same pages for reference

**Train/test split**: For the 10-page pilot, use 8 pages for training and 2 for evaluation. At scale-up, reserve ~10% of pages as held-out test set (pages not used in training data construction at all).

Script: `eval_effocr.py`

## Fork EfficientOCR (Step 0)

Fork `dell-research-harvard/efficient_ocr` to `dangerouspress/efficient_ocr` (or personal GitHub). This is a prerequisite for the whole pipeline.

**Why fork**: The upstream package has stale dependencies (yolov5, `setuptools<70`, Python 3.11 only) and the repo appears unmaintained. A fork lets us modernize without waiting on upstream.

**What to fix in the fork**:
1. **Remove yolov5 dependency for recognizer-only workflows** — the AS pipeline handles detection via ONNX models with `yolov8` backend, so we don't need yolov5 at all. Guard or remove the yolov5 import so it's only loaded if someone explicitly trains a detector.
2. **Replace `pkg_resources` usage with `importlib.metadata`** — eliminates the `setuptools<70` pin
3. **Target Python 3.12+** compatibility
4. **Clean up any other dead dependencies** while we're in there

**AS pipeline detection**: Uses ONNX models (`layout_model_new.onnx`, `line_model_new.onnx`) with `--*_model_backend yolov8` flags. Zero yolov5 imports in the AS pipeline code itself. This is already the working configuration from the original proposal.

**Package management**: Use `uv` throughout — `uv venv`, `uv pip install`, etc.

## Legibility Classifier at Inference Time

The legibility classifier is not fine-tuned (not trainable in-package), but it rejects ~50% of lines at reduced resolution. For inference on reduced-res images, either:
- Lower the legibility threshold to accept more lines
- Disable the classifier entirely and rely on recognizer confidence
- This is an inference-time configuration, not a training concern

## Scripts Summary

| Script | Purpose |
|--------|---------|
| `sample_pages.py` | Select diverse pages from LOC downloads |
| `run_as_pipeline.py` | Wrapper to run AS pipeline on selected JP2s |
| `verify_lines.py` | Send line crops to Qwen3-VL via OpenRouter |
| `build_effocr_training_data.py` | Extract char/word crops at multiple resolutions |
| `finetune_effocr.py` | Run EfficientOCR training |
| `eval_effocr.py` | Evaluate original vs fine-tuned |

## Project Organization

This work lives in the existing `dangerouspress-ocr-finetune` repo alongside the GLM-OCR pipeline. New scripts go in a `scripts/effocr/` subdirectory. Training data and outputs go in a `data/effocr/` subdirectory.

## Infrastructure

- **All local** — no Longleaf for the pilot
- **Python 3.12+** (after fork modernizes efficient_ocr)
- **Package management**: `uv` throughout
- **JP2 support**: `brew install openjpeg` + Pillow or `glymur`
- **AS pipeline**: Needs GPU for YOLO detection (local GPU or CPU with patience)
- **Qwen3-VL**: OpenRouter API (needs `OPENROUTER_API_KEY` env var)
- **EfficientOCR training**: GPU recommended, T4-level sufficient
- **EfficientOCR inference**: CPU

## Scale-Up Plan

After the 10-page pilot validates the pipeline:
1. Strategically sample a few thousand pages across decades/quality levels
2. Run the full pipeline — a few thousand pages yields hundreds of thousands of verified lines
3. Re-evaluate and iterate on training hyperparameters

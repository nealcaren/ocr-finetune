# newspaper-ocr: Modular OCR Pipeline for Historical Newspapers

## Goal

A standalone Python package that provides end-to-end OCR for historical newspaper scans. Modular architecture with swappable detection, recognition, and output formatting backends. Ships with two detectors (AS YOLO, PP-DocLayout), two recognizers (Tesseract, EffOCR), battle-tested newspaper reading order post-processing, and three output formats.

## Architecture

Four-stage pipeline with swappable components:

```
Image (JP2/JPG/PNG, any resolution)
    │
    ▼
┌─────────────┐
│  Detector    │  ← AS YOLO (default), PP-DocLayout, swappable
│  (layout +   │
│   lines)     │
└─────┬───────┘
      │  PageLayout: regions + lines + bounding boxes + crops
      ▼
┌─────────────────┐
│  LayoutProcessor │  ← newspaper reading order, dedup, gap-filling
│  (post-process)  │     Ported from dangerouspress-ocr/ocr_pipeline.py
└─────┬───────────┘
      │  PageLayout: regions reordered, cleaned, merged
      │
      ├──── line-level path ──────┐
      │                           ▼
      │                   ┌──────────────┐
      │                   │LineRecognizer │  ← Tesseract, EffOCR
      │                   └──────────────┘
      │
      └──── region-level path ────┐
                                  ▼
                          ┌──────────────┐
                          │RegionRecognizer│  ← VLM (interface only, future)
                          └──────────────┘
      │
      ▼
┌─────────────┐
│  Formatter   │  ← text, JSON, hOCR
└─────────────┘
```

Each stage has a base class/protocol. The Pipeline class wires them together. Users select backends via config or constructor args. Adding a new backend = one file + one registry entry.

## Detectors

### AS YOLO (default)
- Layout detection (YOLO v8, ONNX) → article regions
- Line detection (YOLO v8, ONNX) → text lines within regions
- Outputs regions AND lines — works directly with LineRecognizer
- Fast, CPU-friendly (~8s per page)

### PP-DocLayout
- PaddleX RT-DETR model (PP-DocLayout_plus-L)
- Detects 20 document layout categories
- Region-level only — does NOT detect individual lines
- When used with a LineRecognizer, each region is treated as a single unit (recognizer handles the full region crop)
- Better at distinguishing content types (headlines, captions, tables, photographs)

## LayoutProcessor (Post-Processing)

Ported from the battle-tested `dangerouspress-ocr/ocr_pipeline.py`. Applied after any detector, before recognition. Six operations:

1. **filter_layout_boxes()** — Keep only >0.5 confidence, relabel non-text types
2. **rescue_low_confidence()** — Recover 0.15–0.5 score detections that fill gaps
3. **deduplicate_boxes()** — Three-pass overlap removal (contained, title-text, near-duplicate)
4. **fill_column_gaps()** — Identify columns via X-midpoint clustering, fill gaps
5. **newspaper_reading_order()** — Column-aware sorting (full-width headers first, then column-by-column, top-to-bottom)
6. **merge_adjacent_blocks()** — Combine vertically adjacent regions (>50% X-overlap, <30px Y-gap, capped at 600px height)

This is optional but on by default. Can be disabled for non-newspaper documents.

## Data Model

```python
@dataclass
class BBox:
    x0: int; y0: int; x1: int; y1: int

@dataclass
class Line:
    bbox: BBox
    image: Image  # PIL crop
    text: str = ""
    confidence: float = 0.0

@dataclass
class Region:
    bbox: BBox
    image: Image  # PIL crop
    label: str  # "article", "headline", etc.
    lines: list[Line]
    text: str = ""  # from lines (LineRecognizer) or direct (RegionRecognizer)

@dataclass
class PageLayout:
    image: Image
    regions: list[Region]
    width: int
    height: int

    @property
    def text(self) -> str:
        return "\n\n".join(r.text for r in self.regions if r.text)
```

`Region.text` can be set by joining `Line.text` values (line-level recognizers) or directly by a VLM (region-level recognizer). Same data model, two paths.

## Interfaces

### Detector

```python
class Detector(ABC):
    @abstractmethod
    def detect(self, image: Image) -> PageLayout:
        """Detect layout regions and lines in an image."""
```

Ships with: `AsYoloDetector` (uses `layout_model_new.onnx` + `line_model_new.onnx`)

### Recognizer

```python
class LineRecognizer(ABC):
    @abstractmethod
    def recognize(self, line: Line) -> Line:
        """Recognize text in a single line crop. Returns Line with text + confidence filled."""

class RegionRecognizer(ABC):
    @abstractmethod
    def recognize(self, region: Region) -> Region:
        """Recognize text in a full region crop. Returns Region with text filled."""
```

Ships with:
- `TesseractRecognizer(LineRecognizer)` — wraps system Tesseract, supports custom traineddata
- `EffocrRecognizer(LineRecognizer)` — wraps forked efficient_ocr with ONNX models

Interface defined but not implemented:
- `VlmRecognizer(RegionRecognizer)` — for future VLM backends (GLM-OCR, Qwen-VL, etc.)

### Formatter

```python
class Formatter(ABC):
    @abstractmethod
    def format(self, layout: PageLayout) -> str:
        """Format OCR results into output string."""
```

Ships with: `TextFormatter`, `JsonFormatter`, `HocrFormatter`

ALTO XML deferred to v2 (complex standard with version fragmentation).

## Pipeline Constructor

```python
Pipeline(
    detector: str = "as_yolo",           # "as_yolo" or "paddlex"
    recognizer: str = "tesseract",       # "tesseract" or "effocr"
    recognizer_model: str | Path | None = None,
    model_cache_dir: str | Path | None = None,    # default: ~/.cache/newspaper-ocr
    layout_processing: bool = True,      # apply reading order + dedup post-processing
    device: str = "cpu",
)
```

**Path resolution:** Pipeline resolves recognizer type from the registry. If recognizer is a `LineRecognizer`, it runs per-line. If it's a `RegionRecognizer` (future VLMs), it runs per-region. No dual-path logic until a `RegionRecognizer` exists — v1 only supports line-level.

**Batch recognition:** `LineRecognizer` also defines `recognize_batch(lines: list[Line]) -> list[Line]` with a default implementation that calls `recognize()` in a loop. `EffocrRecognizer` overrides this for efficient batching.

## User Interface

### Python API

```python
from newspaper_ocr import Pipeline

# Simple — defaults (AS YOLO + Tesseract, plain text)
pipe = Pipeline()
text = pipe.ocr("page.jp2")

# Choose backend
pipe = Pipeline(recognizer="effocr")
result = pipe.ocr("page.jp2", output="json")

# Use fine-tuned models
pipe = Pipeline(
    recognizer="tesseract",
    recognizer_model="path/to/news_gold.traineddata",
)

# Custom cache dir (for HPC with home quota limits)
pipe = Pipeline(model_cache_dir="/work/users/ncaren/cache/newspaper-ocr")

# Batch
results = pipe.ocr_batch(["page1.jp2", "page2.jp2"])
```

### CLI

```bash
# Simple
newspaper-ocr page.jp2

# Options
newspaper-ocr page.jp2 --backend tesseract --output json
newspaper-ocr *.jp2 --backend tesseract --output text --outdir results/

# Fine-tuned model
newspaper-ocr page.jp2 --backend tesseract --model news_gold.traineddata
```

## Package Structure

```
newspaper-ocr/
  src/newspaper_ocr/
    __init__.py              # Public API: Pipeline, models
    pipeline.py              # Pipeline class
    models.py                # BBox, Line, Region, PageLayout
    layout_processor.py      # Reading order, dedup, gap-fill (ported from ocr_pipeline.py)
    registry.py              # Backend registries

    detectors/
      __init__.py            # registry + base class
      base.py                # Detector ABC
      as_yolo.py             # AS YOLO (layout + line detection)
      paddlex.py             # PP-DocLayout_plus-L (region detection, no lines)

    recognizers/
      __init__.py            # registry + base class
      base.py                # LineRecognizer + RegionRecognizer ABCs
      tesseract.py           # Tesseract wrapper
      effocr.py              # EfficientOCR wrapper

    formatters/
      __init__.py            # registry + base class
      base.py                # Formatter ABC
      text.py                # Plain text
      json_fmt.py            # Structured JSON
      hocr.py                # hOCR HTML

    cli.py                   # CLI entry point

  pyproject.toml
  README.md
```

## Dependencies

Core (always installed):
- `pillow` — image handling, JP2 support requires system `openjpeg` library
- `opencv-python-headless` — image loading, resizing, letterboxing for YOLO
- `onnxruntime` — AS YOLO detection models
- `numpy`

Note: `torch`/`torchvision` is needed for NMS in the AS YOLO detector. This is a heavy dependency (~2GB). For v1, accept the tradeoff. For v2, consider replacing with a pure numpy NMS to eliminate the torch requirement.

Optional extras:
- `newspaper-ocr[tesseract]` — requires system `tesseract` binary
- `newspaper-ocr[effocr]` — installs forked `efficient-ocr` (which brings torch anyway)
- `newspaper-ocr[paddlex]` — installs `paddlex` + `paddlepaddle` for PP-DocLayout detector
- `newspaper-ocr[all]` — everything

**JP2 support:** Pillow's JP2 support requires the `openjpeg` system library (`brew install openjpeg` on macOS, `apt install libopenjp2-7` on Ubuntu). If unavailable, JP2 files will raise a clear error message with install instructions.

## Model Distribution

AS YOLO models are required for the default detector:
- `layout_model_new.onnx` (99MB) — layout detection
- `line_model_new.onnx` (43MB) — line detection
- `label_map_layout.json` — class ID → label mapping (bundled in package, <1KB)

**Strategy:** Download on first use from HuggingFace (or GitHub releases), cache to `~/.cache/newspaper-ocr/models/` (configurable via `model_cache_dir`). Show progress bar during download. Too large for PyPI bundling (100MB compressed limit).

## Error Handling

- **Missing Tesseract binary:** `ImportError` with install instructions at `Pipeline` construction time
- **Missing ONNX models:** Auto-download on first use; if download fails, clear error with manual download URL
- **Zero regions detected:** Return `PageLayout` with empty `regions` list (not an error — blank/degraded pages happen)
- **Empty line crops:** Skip silently, log warning
- **Corrupt model file:** `RuntimeError` with clear message
- **JP2 without openjpeg:** `ImportError` with system package install instructions

## Registry Pattern

Each stage type has a registry dict mapping string names to classes:

```python
DETECTORS = {"as_yolo": AsYoloDetector}
LINE_RECOGNIZERS = {"tesseract": TesseractRecognizer, "effocr": EffocrRecognizer}
REGION_RECOGNIZERS = {}  # future VLM backends
FORMATTERS = {"text": TextFormatter, "json": JsonFormatter, "alto": AltoFormatter, "hocr": HocrFormatter}
```

`Pipeline(recognizer="tesseract")` resolves via the registry. Third-party backends can register themselves.

## What This Does NOT Include

- Training/fine-tuning code (stays in `dangerouspress-ocr-finetune`)
- LLM verification pipeline (stays in `dangerouspress-ocr-finetune`)
- Page sampling or dataset construction
- Evaluation scripts

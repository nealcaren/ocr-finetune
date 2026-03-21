# newspaper-ocr: Modular OCR Pipeline for Historical Newspapers

## Goal

A standalone Python package that provides end-to-end OCR for historical newspaper scans. Modular architecture with swappable detection, recognition, and output formatting backends. Ships with AS YOLO detection, Tesseract and EfficientOCR recognition, and four output formats.

## Architecture

Three-stage pipeline with swappable components:

```
Image (JP2/JPG/PNG, any resolution)
    │
    ▼
┌─────────────┐
│  Detector    │  ← AS YOLO (default), swappable
│  (layout +   │
│   lines)     │
└─────┬───────┘
      │  PageLayout: regions + lines + bounding boxes + crops
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
│  Formatter   │  ← text, JSON, ALTO XML, hOCR
└─────────────┘
```

Each stage has a base class/protocol. The Pipeline class wires them together. Users select backends via config or constructor args. Adding a new backend = one file + one registry entry.

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

Ships with: `TextFormatter`, `JsonFormatter`, `AltoFormatter`, `HocrFormatter`

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
    __init__.py              # Pipeline export
    pipeline.py              # Pipeline class
    models.py                # BBox, Line, Region, PageLayout

    detectors/
      __init__.py            # registry + base class
      base.py                # Detector ABC
      as_yolo.py             # AS YOLO (layout + line detection)

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
      alto.py                # ALTO XML
      hocr.py                # hOCR HTML

    cli.py                   # CLI entry point

  pyproject.toml
  README.md
```

## Dependencies

Core (always installed):
- `pillow` — image handling
- `onnxruntime` — AS YOLO detection models
- `torch` — NMS, tensor ops for detection
- `numpy`

Optional extras:
- `newspaper-ocr[tesseract]` — requires system `tesseract` binary
- `newspaper-ocr[effocr]` — installs forked `efficient-ocr`
- `newspaper-ocr[all]` — everything

## Model Distribution

AS YOLO models (`layout_model_new.onnx`, `line_model_new.onnx`) are required for the default detector. Options:
- Bundle in the package (adds ~140MB)
- Download on first use from HuggingFace/GitHub releases
- User provides path

Recommend: download on first use with a progress bar, cache locally.

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

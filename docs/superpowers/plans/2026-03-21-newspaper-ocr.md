# newspaper-ocr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python package for end-to-end OCR of historical newspaper scans, with swappable detection, recognition, and formatting backends.

**Architecture:** Four-stage pipeline (Detector → LayoutProcessor → Recognizer → Formatter) with registry-based backend selection. Each stage has an ABC. The Pipeline class wires them together. Ships with AS YOLO + PP-DocLayout detection, newspaper reading order post-processing, Tesseract + EffOCR recognition, and text/JSON/hOCR output.

**Tech Stack:** Python 3.11+, pillow, opencv-python-headless, onnxruntime, torch/torchvision (for NMS), numpy, click (CLI)

**Spec:** `docs/superpowers/specs/2026-03-21-newspaper-ocr-design.md`

**Repo:** New repo at `/Users/nealcaren/Documents/GitHub/newspaper-ocr`

---

## File Structure

```
newspaper-ocr/
  src/newspaper_ocr/
    __init__.py              # Public API: Pipeline, models
    pipeline.py              # Pipeline class
    models.py                # BBox, Line, Region, PageLayout dataclasses
    registry.py              # Backend registries
    layout_processor.py      # Reading order, dedup, gap-fill (ported from ocr_pipeline.py)

    detectors/
      __init__.py            # Detector base + registry access
      base.py                # Detector ABC
      as_yolo.py             # AS YOLO implementation (~300 lines, ported from run_as_extraction.py)
      paddlex.py             # PP-DocLayout_plus-L (region detection, no lines)

    recognizers/
      __init__.py            # Recognizer base + registry access
      base.py                # LineRecognizer + RegionRecognizer ABCs
      tesseract.py           # Tesseract subprocess wrapper
      effocr.py              # EfficientOCR wrapper

    formatters/
      __init__.py            # Formatter base + registry access
      base.py                # Formatter ABC
      text.py                # Plain text
      json_fmt.py            # Structured JSON
      hocr.py                # hOCR HTML

    cli.py                   # Click CLI entry point

  tests/
    conftest.py              # Shared fixtures (sample images, mock models)
    test_models.py           # Data model tests
    test_pipeline.py         # Pipeline integration tests
    test_registry.py         # Registry tests
    test_detectors/
      test_as_yolo.py        # AS YOLO detector tests
    test_recognizers/
      test_tesseract.py      # Tesseract recognizer tests
      test_effocr.py         # EffOCR recognizer tests
    test_formatters/
      test_text.py
      test_json.py
      test_hocr.py
    test_cli.py

  pyproject.toml
  README.md
```

---

### Task 1: Repo Setup and Data Model

**Files:**
- Create: `pyproject.toml`
- Create: `src/newspaper_ocr/__init__.py`
- Create: `src/newspaper_ocr/models.py`
- Create: `src/newspaper_ocr/registry.py`
- Create: `tests/conftest.py`
- Create: `tests/test_models.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Create the repo**

```bash
mkdir -p /Users/nealcaren/Documents/GitHub/newspaper-ocr
cd /Users/nealcaren/Documents/GitHub/newspaper-ocr
git init
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "newspaper-ocr"
version = "0.1.0"
description = "Modular OCR pipeline for historical newspaper scans"
requires-python = ">=3.11"
dependencies = [
    "pillow>=10.0",
    "opencv-python-headless>=4.8",
    "onnxruntime>=1.16",
    "numpy>=1.24",
    "torch>=2.0",
    "torchvision>=0.15",
    "click>=8.0",
    "huggingface-hub>=0.20",
]

[project.optional-dependencies]
tesseract = []  # requires system tesseract binary
effocr = ["efficient-ocr @ git+https://github.com/nealcaren/efficient_ocr.git"]
all = ["newspaper-ocr[tesseract,effocr]"]
dev = ["pytest>=7.0", "pytest-cov"]

[project.scripts]
newspaper-ocr = "newspaper_ocr.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write data model tests**

```python
# tests/test_models.py
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np


def test_bbox_creation():
    bbox = BBox(x0=10, y0=20, x1=100, y1=50)
    assert bbox.x0 == 10
    assert bbox.width == 90
    assert bbox.height == 30


def test_line_default_text():
    img = Image.fromarray(np.zeros((30, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(0, 0, 200, 30), image=img)
    assert line.text == ""
    assert line.confidence == 0.0


def test_region_text_from_lines():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    lines = [
        Line(bbox=BBox(0, 0, 200, 30), image=img, text="Hello"),
        Line(bbox=BBox(0, 30, 200, 60), image=img, text="world"),
    ]
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", lines=lines)
    assert region.text == ""  # text not auto-assembled; Pipeline does this


def test_page_layout_text():
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    r1 = Region(bbox=BBox(0, 0, 400, 200), image=img, label="article", lines=[], text="First paragraph.")
    r2 = Region(bbox=BBox(0, 200, 400, 500), image=img, label="article", lines=[], text="Second paragraph.")
    layout = PageLayout(image=img, regions=[r1, r2], width=400, height=500)
    assert layout.text == "First paragraph.\n\nSecond paragraph."
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /Users/nealcaren/Documents/GitHub/newspaper-ocr
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/test_models.py -v
```

Expected: ImportError (models.py doesn't exist yet)

- [ ] **Step 5: Implement `src/newspaper_ocr/models.py`**

```python
"""Core data models for the OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from PIL import Image


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Line:
    bbox: BBox
    image: Image.Image
    text: str = ""
    confidence: float = 0.0


@dataclass
class Region:
    bbox: BBox
    image: Image.Image
    label: str
    lines: list[Line] = field(default_factory=list)
    text: str = ""


@dataclass
class PageLayout:
    image: Image.Image
    regions: list[Region] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(r.text for r in self.regions if r.text)
```

- [ ] **Step 6: Write registry tests**

```python
# tests/test_registry.py
from newspaper_ocr.registry import Registry


def test_register_and_get():
    reg = Registry("test")
    reg.register("foo", str)
    assert reg.get("foo") is str


def test_get_unknown_raises():
    reg = Registry("test")
    import pytest
    with pytest.raises(KeyError, match="Unknown test"):
        reg.get("bar")


def test_list_registered():
    reg = Registry("test")
    reg.register("a", int)
    reg.register("b", str)
    assert reg.list() == ["a", "b"]
```

- [ ] **Step 7: Implement `src/newspaper_ocr/registry.py`**

```python
"""Simple registry for backend discovery."""

from __future__ import annotations


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, type] = {}

    def register(self, key: str, cls: type) -> None:
        self._entries[key] = cls

    def get(self, key: str) -> type:
        if key not in self._entries:
            raise KeyError(f"Unknown {self.name}: '{key}'. Available: {self.list()}")
        return self._entries[key]

    def list(self) -> list[str]:
        return sorted(self._entries.keys())
```

- [ ] **Step 8: Create `src/newspaper_ocr/__init__.py`**

```python
"""newspaper-ocr: Modular OCR pipeline for historical newspaper scans."""

from newspaper_ocr.models import BBox, Line, Region, PageLayout

__all__ = ["BBox", "Line", "Region", "PageLayout"]
```

- [ ] **Step 9: Create empty `tests/conftest.py`**

```python
"""Shared test fixtures."""
```

- [ ] **Step 10: Run tests**

```bash
pytest tests/test_models.py tests/test_registry.py -v
```

Expected: All pass.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: init repo with data model and registry"
```

---

### Task 2: Base Classes and Pipeline Skeleton

**Files:**
- Create: `src/newspaper_ocr/detectors/__init__.py`
- Create: `src/newspaper_ocr/detectors/base.py`
- Create: `src/newspaper_ocr/recognizers/__init__.py`
- Create: `src/newspaper_ocr/recognizers/base.py`
- Create: `src/newspaper_ocr/formatters/__init__.py`
- Create: `src/newspaper_ocr/formatters/base.py`
- Create: `src/newspaper_ocr/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write pipeline test with mock backends**

```python
# tests/test_pipeline.py
from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.recognizers.base import LineRecognizer
from newspaper_ocr.formatters.base import Formatter
from PIL import Image
import numpy as np


class MockDetector(Detector):
    def detect(self, image):
        w, h = image.size
        line_img = image.crop((0, 0, w, 30))
        line = Line(bbox=BBox(0, 0, w, 30), image=line_img)
        region = Region(bbox=BBox(0, 0, w, h), image=image, label="article", lines=[line])
        return PageLayout(image=image, regions=[region], width=w, height=h)


class MockRecognizer(LineRecognizer):
    def recognize(self, line):
        line.text = "mock text"
        line.confidence = 0.99
        return line


class MockFormatter(Formatter):
    def format(self, layout):
        return layout.text


def test_pipeline_end_to_end():
    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        formatter=MockFormatter(),
    )
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    result = pipe.run(img)
    assert "mock text" in result


def test_pipeline_ocr_from_path(tmp_path):
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    img_path = tmp_path / "test.png"
    img.save(str(img_path))

    pipe = Pipeline(
        detector=MockDetector(),
        recognizer=MockRecognizer(),
        formatter=MockFormatter(),
    )
    result = pipe.ocr(str(img_path))
    assert "mock text" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement base classes**

```python
# src/newspaper_ocr/detectors/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image
from newspaper_ocr.models import PageLayout


class Detector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> PageLayout:
        """Detect layout regions and lines in an image."""
```

```python
# src/newspaper_ocr/recognizers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from newspaper_ocr.models import Line, Region


class LineRecognizer(ABC):
    @abstractmethod
    def recognize(self, line: Line) -> Line:
        """Recognize text in a single line crop."""

    def recognize_batch(self, lines: list[Line]) -> list[Line]:
        """Recognize text in multiple lines. Override for efficiency."""
        return [self.recognize(line) for line in lines]


class RegionRecognizer(ABC):
    @abstractmethod
    def recognize(self, region: Region) -> Region:
        """Recognize text in a full region crop."""
```

```python
# src/newspaper_ocr/formatters/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from newspaper_ocr.models import PageLayout


class Formatter(ABC):
    @abstractmethod
    def format(self, layout: PageLayout) -> str:
        """Format OCR results into output string."""
```

- [ ] **Step 4: Implement Pipeline**

```python
# src/newspaper_ocr/pipeline.py
from __future__ import annotations

from pathlib import Path
from PIL import Image

from newspaper_ocr.models import PageLayout
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.formatters.base import Formatter


class Pipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        recognizer: LineRecognizer | RegionRecognizer | None = None,
        formatter: Formatter | None = None,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.formatter = formatter

    def run(self, image: Image.Image) -> str:
        """Run the full pipeline on a PIL Image."""
        layout = self.detector.detect(image)

        # Post-process layout (reading order, dedup, etc.)
        if hasattr(self, 'layout_processor'):
            layout = self.layout_processor.process(layout)

        if isinstance(self.recognizer, LineRecognizer):
            for region in layout.regions:
                region.lines = self.recognizer.recognize_batch(region.lines)
                region.text = " ".join(line.text for line in region.lines if line.text)
        elif isinstance(self.recognizer, RegionRecognizer):
            for region in layout.regions:
                region = self.recognizer.recognize(region)

        return self.formatter.format(layout)

    def ocr(self, path: str | Path, output: str | None = None) -> str:
        """Run OCR on an image file."""
        image = Image.open(str(path)).convert("RGB")
        return self.run(image)

    def ocr_batch(self, paths: list[str | Path]) -> list[str]:
        """Run OCR on multiple image files."""
        return [self.ocr(p) for p in paths]
```

- [ ] **Step 5: Create `__init__.py` files for subpackages**

```python
# src/newspaper_ocr/detectors/__init__.py
from newspaper_ocr.detectors.base import Detector

# src/newspaper_ocr/recognizers/__init__.py
from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer

# src/newspaper_ocr/formatters/__init__.py
from newspaper_ocr.formatters.base import Formatter
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/ -v
```

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: add base classes and pipeline skeleton with mock tests"
```

---

### Task 3: Formatters (Text, JSON, hOCR)

**Files:**
- Create: `src/newspaper_ocr/formatters/text.py`
- Create: `src/newspaper_ocr/formatters/json_fmt.py`
- Create: `src/newspaper_ocr/formatters/hocr.py`
- Create: `tests/test_formatters/test_text.py`
- Create: `tests/test_formatters/test_json.py`
- Create: `tests/test_formatters/test_hocr.py`

- [ ] **Step 1: Write formatter tests**

```python
# tests/test_formatters/test_text.py
from newspaper_ocr.formatters.text import TextFormatter
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np

def _make_layout():
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    r1 = Region(bbox=BBox(0, 0, 400, 200), image=img, label="headline", lines=[], text="BIG NEWS TODAY")
    r2 = Region(bbox=BBox(0, 200, 400, 500), image=img, label="article", lines=[], text="Something happened.")
    return PageLayout(image=img, regions=[r1, r2], width=400, height=500)

def test_text_formatter():
    result = TextFormatter().format(_make_layout())
    assert "BIG NEWS TODAY" in result
    assert "Something happened." in result
```

```python
# tests/test_formatters/test_json.py
import json
from newspaper_ocr.formatters.json_fmt import JsonFormatter
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np

def test_json_formatter():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(0, 0, 200, 30), image=img, text="Hello", confidence=0.95)
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", lines=[line], text="Hello")
    layout = PageLayout(image=img, regions=[region], width=200, height=100)
    result = json.loads(JsonFormatter().format(layout))
    assert result["width"] == 200
    assert result["regions"][0]["label"] == "article"
    assert result["regions"][0]["lines"][0]["text"] == "Hello"
    assert result["regions"][0]["lines"][0]["confidence"] == 0.95
    assert result["regions"][0]["lines"][0]["bbox"] == {"x0": 0, "y0": 0, "x1": 200, "y1": 30}
```

```python
# tests/test_formatters/test_hocr.py
from newspaper_ocr.formatters.hocr import HocrFormatter
from newspaper_ocr.models import BBox, Line, Region, PageLayout
from PIL import Image
import numpy as np

def test_hocr_has_structure():
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    line = Line(bbox=BBox(10, 20, 190, 50), image=img, text="Hello world")
    region = Region(bbox=BBox(0, 0, 200, 100), image=img, label="article", lines=[line], text="Hello world")
    layout = PageLayout(image=img, regions=[region], width=200, height=100)
    result = HocrFormatter().format(layout)
    assert 'class="ocr_page"' in result
    assert 'class="ocr_carea"' in result
    assert 'class="ocr_line"' in result
    assert "Hello world" in result
    assert "bbox 10 20 190 50" in result
```

- [ ] **Step 2: Implement formatters**

```python
# src/newspaper_ocr/formatters/text.py
from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class TextFormatter(Formatter):
    def format(self, layout: PageLayout) -> str:
        return layout.text
```

```python
# src/newspaper_ocr/formatters/json_fmt.py
import json
from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class JsonFormatter(Formatter):
    def format(self, layout: PageLayout) -> str:
        data = {
            "width": layout.width,
            "height": layout.height,
            "regions": [
                {
                    "label": r.label,
                    "bbox": {"x0": r.bbox.x0, "y0": r.bbox.y0, "x1": r.bbox.x1, "y1": r.bbox.y1},
                    "text": r.text,
                    "lines": [
                        {
                            "text": line.text,
                            "confidence": line.confidence,
                            "bbox": {"x0": line.bbox.x0, "y0": line.bbox.y0, "x1": line.bbox.x1, "y1": line.bbox.y1},
                        }
                        for line in r.lines
                    ],
                }
                for r in layout.regions
            ],
        }
        return json.dumps(data, indent=2)
```

```python
# src/newspaper_ocr/formatters/hocr.py
from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class HocrFormatter(Formatter):
    def format(self, layout: PageLayout) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            "<head><title>OCR Output</title></head>",
            "<body>",
            f'  <div class="ocr_page" title="bbox 0 0 {layout.width} {layout.height}">',
        ]
        for i, region in enumerate(layout.regions):
            rb = region.bbox
            lines.append(f'    <div class="ocr_carea" id="block_{i}" title="bbox {rb.x0} {rb.y0} {rb.x1} {rb.y1}">')
            for j, line in enumerate(region.lines):
                lb = line.bbox
                lines.append(f'      <span class="ocr_line" id="line_{i}_{j}" title="bbox {lb.x0} {lb.y0} {lb.x1} {lb.y1}">{_escape(line.text)}</span>')
            lines.append("    </div>")
        lines.append("  </div>")
        lines.append("</body>")
        lines.append("</html>")
        return "\n".join(lines)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
```

- [ ] **Step 3: Register formatters**

Update `src/newspaper_ocr/formatters/__init__.py`:
```python
from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.formatters.text import TextFormatter
from newspaper_ocr.formatters.json_fmt import JsonFormatter
from newspaper_ocr.formatters.hocr import HocrFormatter
from newspaper_ocr.registry import Registry

FORMATTERS = Registry("formatter")
FORMATTERS.register("text", TextFormatter)
FORMATTERS.register("json", JsonFormatter)
FORMATTERS.register("hocr", HocrFormatter)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add text, JSON, and hOCR formatters"
```

---

### Task 4: Tesseract Recognizer

**Files:**
- Create: `src/newspaper_ocr/recognizers/tesseract.py`
- Create: `tests/test_recognizers/test_tesseract.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_recognizers/test_tesseract.py
import subprocess
import pytest
from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
from newspaper_ocr.models import BBox, Line
from PIL import Image, ImageDraw, ImageFont


def _make_line_image(text="Hello World"):
    """Create a simple line image with text for testing."""
    img = Image.new("RGB", (400, 50), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill="black")
    return img


@pytest.fixture
def has_tesseract():
    result = subprocess.run(["tesseract", "--version"], capture_output=True)
    if result.returncode != 0:
        pytest.skip("tesseract not installed")


def test_tesseract_recognizes_text(has_tesseract):
    img = _make_line_image("Hello World")
    line = Line(bbox=BBox(0, 0, 400, 50), image=img)
    rec = TesseractRecognizer()
    result = rec.recognize(line)
    # Tesseract should get something close (exact match not guaranteed with default font)
    assert len(result.text) > 0
    assert result.confidence > 0


def test_tesseract_custom_model(has_tesseract):
    rec = TesseractRecognizer(model="eng")
    img = _make_line_image("Test")
    line = Line(bbox=BBox(0, 0, 400, 50), image=img)
    result = rec.recognize(line)
    assert len(result.text) > 0


def test_tesseract_batch(has_tesseract):
    lines = [
        Line(bbox=BBox(0, 0, 400, 50), image=_make_line_image("First")),
        Line(bbox=BBox(0, 0, 400, 50), image=_make_line_image("Second")),
    ]
    rec = TesseractRecognizer()
    results = rec.recognize_batch(lines)
    assert len(results) == 2
    assert all(r.text for r in results)
```

- [ ] **Step 2: Implement Tesseract recognizer**

```python
# src/newspaper_ocr/recognizers/tesseract.py
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from newspaper_ocr.models import Line
from newspaper_ocr.recognizers.base import LineRecognizer


class TesseractRecognizer(LineRecognizer):
    def __init__(
        self,
        model: str = "eng",
        tessdata_dir: str | Path | None = None,
    ):
        self.model = model
        self.tessdata_dir = str(tessdata_dir) if tessdata_dir else None
        self._check_installed()

    def _check_installed(self):
        try:
            subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            raise ImportError(
                "Tesseract is not installed. Install with:\n"
                "  macOS: brew install tesseract\n"
                "  Ubuntu: apt install tesseract-ocr\n"
                "  See: https://github.com/tesseract-ocr/tesseract"
            )

    def recognize(self, line: Line) -> Line:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            line.image.save(f.name)
            cmd = ["tesseract", f.name, "stdout", "--psm", "7"]
            if self.tessdata_dir:
                cmd.extend(["--tessdata-dir", self.tessdata_dir])
            cmd.extend(["-l", self.model])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            Path(f.name).unlink(missing_ok=True)

        line.text = result.stdout.strip()
        # Tesseract doesn't give per-line confidence easily; default to 1.0 if text was produced
        line.confidence = 0.9 if line.text else 0.0
        return line
```

- [ ] **Step 3: Register**

Update `src/newspaper_ocr/recognizers/__init__.py`:
```python
from newspaper_ocr.recognizers.base import LineRecognizer, RegionRecognizer
from newspaper_ocr.registry import Registry

RECOGNIZERS = Registry("recognizer")

try:
    from newspaper_ocr.recognizers.tesseract import TesseractRecognizer
    RECOGNIZERS.register("tesseract", TesseractRecognizer)
except ImportError:
    pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tesseract line recognizer"
```

---

### Task 5: AS YOLO Detector

**Files:**
- Create: `src/newspaper_ocr/detectors/as_yolo.py`
- Create: `tests/test_detectors/test_as_yolo.py`

This is the largest task — porting the detection code from `run_as_extraction.py` in the fine-tuning repo. The key functions to port: `letterbox()`, `run_layout_detection()`, `run_line_detection()`, `readjust_line_detections()`.

- [ ] **Step 1: Write integration test**

```python
# tests/test_detectors/test_as_yolo.py
import pytest
from pathlib import Path
from newspaper_ocr.detectors.as_yolo import AsYoloDetector
from PIL import Image
import numpy as np


@pytest.fixture
def has_models():
    """Skip if ONNX models aren't available locally."""
    model_dir = Path.home() / ".cache" / "newspaper-ocr" / "models"
    if not (model_dir / "layout_model_new.onnx").exists():
        # Also check Dropbox location
        alt = Path("/Users/nealcaren/Dropbox/american-stories/american_stories_models")
        if not (alt / "layout_model_new.onnx").exists():
            pytest.skip("AS YOLO models not available")


def test_detect_returns_layout(has_models):
    """Test that detection produces a PageLayout with regions and lines."""
    detector = AsYoloDetector()
    # Create a simple test image (won't have real text, but should not crash)
    img = Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8))
    layout = detector.detect(img)
    assert layout.width == 600
    assert layout.height == 800
    assert isinstance(layout.regions, list)
    # Random noise may or may not produce detections — just check it doesn't crash


def test_detect_real_image(has_models):
    """Test on an actual newspaper scan if available."""
    test_img = Path("/Volumes/Lightning/chronicling-america/loc_downloads/sn84025908/1856-08-30/seq-4.jp2")
    if not test_img.exists():
        pytest.skip("Test JP2 not available")
    detector = AsYoloDetector()
    img = Image.open(test_img)
    layout = detector.detect(img)
    assert len(layout.regions) > 0
    total_lines = sum(len(r.lines) for r in layout.regions)
    assert total_lines > 100  # This page should have ~1100 lines
```

- [ ] **Step 2: Port AS YOLO detection code**

Create `src/newspaper_ocr/detectors/as_yolo.py`. This should be ported from `/Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune/scripts/effocr/run_as_extraction.py`, extracting the detection logic into a clean class.

Key components to port:
- `letterbox()` — image preprocessing for YOLO
- `run_layout_detection()` — layout region detection
- `run_line_detection()` — line detection within regions (with chunk-and-merge for tall regions)
- `readjust_line_detections()` — NMS and coordinate back-projection
- Model loading and caching

The constructor should accept model paths or auto-download:

```python
class AsYoloDetector(Detector):
    def __init__(
        self,
        model_dir: str | Path | None = None,
        layout_model: str | Path | None = None,
        line_model: str | Path | None = None,
    ):
        # Resolve model paths: explicit > model_dir > auto-download
        ...
```

This file will be ~300 lines. Port carefully from `run_as_extraction.py`, adapting to use the `PageLayout`/`Region`/`Line` data models instead of raw dicts.

- [ ] **Step 3: Register**

Update `src/newspaper_ocr/detectors/__init__.py`:
```python
from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.registry import Registry

DETECTORS = Registry("detector")

try:
    from newspaper_ocr.detectors.as_yolo import AsYoloDetector
    DETECTORS.register("as_yolo", AsYoloDetector)
except ImportError:
    pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add AS YOLO detector (layout + line detection)"
```

---

### Task 5b: LayoutProcessor (Reading Order + Post-Processing)

**Files:**
- Create: `src/newspaper_ocr/layout_processor.py`
- Create: `tests/test_layout_processor.py`

Port the battle-tested post-processing from `/Users/nealcaren/Documents/GitHub/dangerouspress-ocr/ocr_pipeline.py`. This applies after any detector, before recognition.

- [ ] **Step 1: Write tests**

```python
# tests/test_layout_processor.py
from newspaper_ocr.layout_processor import LayoutProcessor
from newspaper_ocr.models import BBox, Region, PageLayout
from PIL import Image
import numpy as np


def _make_region(x0, y0, x1, y1, label="text", confidence=0.9):
    img = Image.fromarray(np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8))
    r = Region(bbox=BBox(x0, y0, x1, y1), image=img, label=label, lines=[])
    r.confidence = confidence
    return r


def test_deduplicates_overlapping_regions():
    """Two regions with 80%+ overlap → keep higher confidence."""
    r1 = _make_region(0, 0, 200, 100, confidence=0.9)
    r2 = _make_region(10, 5, 195, 95, confidence=0.7)
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    layout = PageLayout(image=img, regions=[r1, r2], width=400, height=500)
    proc = LayoutProcessor()
    result = proc.process(layout)
    assert len(result.regions) == 1
    assert result.regions[0].confidence == 0.9


def test_reading_order_columns():
    """Regions in two columns should be ordered: left col top-to-bottom, then right col."""
    left_top = _make_region(0, 0, 180, 200)
    left_bot = _make_region(0, 200, 180, 400)
    right_top = _make_region(220, 0, 400, 200)
    right_bot = _make_region(220, 200, 400, 400)
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    layout = PageLayout(image=img, regions=[right_bot, left_top, right_top, left_bot], width=400, height=500)
    proc = LayoutProcessor()
    result = proc.process(layout)
    # Left column first, then right
    xs = [r.bbox.x0 for r in result.regions]
    assert xs == [0, 0, 220, 220]


def test_filters_low_confidence():
    """Regions below threshold are removed."""
    high = _make_region(0, 0, 200, 100, confidence=0.8)
    low = _make_region(0, 100, 200, 200, confidence=0.1)
    img = Image.fromarray(np.zeros((500, 400, 3), dtype=np.uint8))
    layout = PageLayout(image=img, regions=[high, low], width=400, height=500)
    proc = LayoutProcessor()
    result = proc.process(layout)
    assert len(result.regions) == 1
```

- [ ] **Step 2: Port LayoutProcessor from ocr_pipeline.py**

Read `/Users/nealcaren/Documents/GitHub/dangerouspress-ocr/ocr_pipeline.py` and port the key functions:
- `filter_layout_boxes()`
- `rescue_low_confidence()`
- `deduplicate_boxes()`
- `fill_column_gaps()`
- `newspaper_reading_order()`
- `merge_adjacent_blocks()`

Adapt them to work with our `PageLayout`/`Region` data model instead of raw dicts. The `LayoutProcessor` class wraps all six steps:

```python
class LayoutProcessor:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def process(self, layout: PageLayout) -> PageLayout:
        if not self.enabled:
            return layout
        # Apply six steps in order
        ...
```

Note: `Region` needs a `confidence` attribute added to `models.py`.

- [ ] **Step 3: Add `confidence` to Region model**

In `models.py`, add `confidence: float = 0.0` to the `Region` dataclass.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_layout_processor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LayoutProcessor with reading order and post-processing"
```

---

### Task 5c: PP-DocLayout Detector

**Files:**
- Create: `src/newspaper_ocr/detectors/paddlex.py`
- Create: `tests/test_detectors/test_paddlex.py`

PP-DocLayout uses PaddleX (RT-DETR architecture). It detects 20 layout categories but does NOT detect individual lines — it outputs region-level bounding boxes only.

- [ ] **Step 1: Write test**

```python
# tests/test_detectors/test_paddlex.py
import pytest
from newspaper_ocr.detectors.paddlex import PaddleXDetector
from PIL import Image
import numpy as np


@pytest.fixture
def has_paddlex():
    try:
        import paddlex
    except ImportError:
        pytest.skip("paddlex not installed")


def test_detect_returns_layout(has_paddlex):
    detector = PaddleXDetector()
    img = Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8))
    layout = detector.detect(img)
    assert layout.width == 600
    assert layout.height == 800
    assert isinstance(layout.regions, list)
    # Regions will have no lines (PaddleX doesn't do line detection)
    for r in layout.regions:
        assert r.lines == []
```

- [ ] **Step 2: Implement PaddleX detector**

```python
# src/newspaper_ocr/detectors/paddlex.py
from __future__ import annotations

from pathlib import Path

from newspaper_ocr.detectors.base import Detector
from newspaper_ocr.models import BBox, Region, PageLayout


class PaddleXDetector(Detector):
    """PP-DocLayout_plus-L detector using PaddleX."""

    # Labels from PP-DocLayout that contain text
    TEXT_LABELS = {"text", "paragraph_title", "doc_title", "figure_title",
                   "reference", "header", "footer", "abstract"}

    def __init__(self, model_name: str = "PP-DocLayout_plus-L", **kwargs):
        try:
            import paddlex
        except ImportError:
            raise ImportError(
                "paddlex is not installed. Install with:\n"
                "  pip install 'newspaper-ocr[paddlex]'\n"
                "  Requires: pip install paddlepaddle paddlex"
            )
        self.model = paddlex.create_model(model_name)

    def detect(self, image):
        import tempfile
        from PIL import Image as PILImage

        w, h = image.size

        # PaddleX expects a file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f.name)
            results = list(self.model.predict(f.name))
            Path(f.name).unlink(missing_ok=True)

        regions = []
        for result in results:
            for box in result.get("boxes", []):
                label = box.get("label", "")
                score = box.get("score", 0.0)
                coords = box.get("coordinate", [0, 0, 0, 0])
                x0, y0, x1, y1 = [int(c) for c in coords]

                region_img = image.crop((x0, y0, x1, y1))
                region = Region(
                    bbox=BBox(x0, y0, x1, y1),
                    image=region_img,
                    label=label,
                    lines=[],  # PaddleX doesn't detect lines
                    confidence=score,
                )
                regions.append(region)

        return PageLayout(image=image, regions=regions, width=w, height=h)
```

- [ ] **Step 3: Register**

Add to `src/newspaper_ocr/detectors/__init__.py`:
```python
try:
    from newspaper_ocr.detectors.paddlex import PaddleXDetector
    DETECTORS.register("paddlex", PaddleXDetector)
except ImportError:
    pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add PP-DocLayout detector via PaddleX"
```

---

### Task 6: EffOCR Recognizer

**Files:**
- Create: `src/newspaper_ocr/recognizers/effocr.py`
- Create: `tests/test_recognizers/test_effocr.py`

- [ ] **Step 1: Write test**

```python
# tests/test_recognizers/test_effocr.py
import pytest
from newspaper_ocr.recognizers.effocr import EffocrRecognizer
from newspaper_ocr.models import BBox, Line
from PIL import Image
from pathlib import Path


@pytest.fixture
def has_effocr():
    try:
        from efficient_ocr import EffOCR
    except ImportError:
        pytest.skip("efficient_ocr not installed")


@pytest.fixture
def has_models():
    # Check if ONNX models exist
    model_dir = Path("/Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune/data/effocr/models")
    if not model_dir.exists():
        pytest.skip("EffOCR models not available")
    return model_dir


def test_effocr_recognize(has_effocr, has_models):
    rec = EffocrRecognizer(model_dir=has_models)
    # Use a real line crop if available
    test_crop = Path("/Users/nealcaren/Documents/GitHub/dangerouspress-ocr-finetune/data/effocr/as_extractions/sn84025908_1856-08-30_seq-4/lines/line_0010.png")
    if not test_crop.exists():
        pytest.skip("Test line crop not available")
    img = Image.open(test_crop)
    line = Line(bbox=BBox(0, 0, img.width, img.height), image=img)
    result = rec.recognize(line)
    assert len(result.text) > 0
```

- [ ] **Step 2: Implement EffOCR recognizer**

```python
# src/newspaper_ocr/recognizers/effocr.py
from __future__ import annotations

import tempfile
from pathlib import Path

from newspaper_ocr.models import Line
from newspaper_ocr.recognizers.base import LineRecognizer


class EffocrRecognizer(LineRecognizer):
    def __init__(
        self,
        model_dir: str | Path | None = None,
    ):
        try:
            from efficient_ocr import EffOCR
        except ImportError:
            raise ImportError(
                "efficient_ocr is not installed. Install with:\n"
                "  pip install 'newspaper-ocr[effocr]'\n"
                "  or: pip install git+https://github.com/nealcaren/efficient_ocr.git"
            )

        model_dir = str(model_dir) if model_dir else "./models"
        self.effocr = EffOCR(config={
            "Global": {"skip_line_detection": True},
            "Recognizer": {
                "char": {
                    "model_backend": "onnx",
                    "model_dir": model_dir,
                    "hf_repo_id": "dell-research-harvard/effocr_en/char_recognizer",
                },
                "word": {
                    "model_backend": "onnx",
                    "model_dir": model_dir,
                    "hf_repo_id": "dell-research-harvard/effocr_en/word_recognizer",
                },
            },
            "Localizer": {
                "model_backend": "onnx",
                "model_dir": model_dir,
                "hf_repo_id": "dell-research-harvard/effocr_en",
            },
            "Line": {
                "model_backend": "onnx",
                "model_dir": model_dir,
                "hf_repo_id": "dell-research-harvard/effocr_en",
            },
        })

    def recognize(self, line: Line) -> Line:
        # Save line image to temp file (EffOCR expects file paths)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            line.image.save(f.name)
            try:
                results = self.effocr.infer(f.name)
                if results and results[0].text:
                    line.text = results[0].text.strip()
                    line.confidence = 0.8  # EffOCR doesn't expose per-line confidence
            except Exception:
                line.text = ""
                line.confidence = 0.0
            finally:
                Path(f.name).unlink(missing_ok=True)
        return line
```

- [ ] **Step 3: Register**

Add to `src/newspaper_ocr/recognizers/__init__.py`:
```python
try:
    from newspaper_ocr.recognizers.effocr import EffocrRecognizer
    RECOGNIZERS.register("effocr", EffocrRecognizer)
except ImportError:
    pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add EfficientOCR line recognizer"
```

---

### Task 7: Pipeline Factory with String-Based Config

**Files:**
- Modify: `src/newspaper_ocr/pipeline.py`
- Modify: `src/newspaper_ocr/__init__.py`
- Create: `tests/test_pipeline_factory.py`

Wire up the registries so `Pipeline(recognizer="tesseract")` works.

- [ ] **Step 1: Write test**

```python
# tests/test_pipeline_factory.py
import pytest
from newspaper_ocr import Pipeline


def test_pipeline_from_strings():
    """Test that Pipeline can be constructed with string backend names."""
    # This will fail if tesseract isn't installed, which is fine
    try:
        pipe = Pipeline(detector="as_yolo", recognizer="tesseract", output="text")
    except (ImportError, KeyError):
        pytest.skip("Required backends not available")
    assert pipe.detector is not None
    assert pipe.recognizer is not None
    assert pipe.formatter is not None


def test_pipeline_default():
    """Test default pipeline construction."""
    try:
        pipe = Pipeline()
    except ImportError:
        pytest.skip("Default backends not available")
    assert pipe.detector is not None
```

- [ ] **Step 2: Update Pipeline to support string config**

Add a class method or update `__init__`:

```python
# In pipeline.py — update __init__ to accept strings
def __init__(
    self,
    detector: Detector | str = "as_yolo",
    recognizer: LineRecognizer | RegionRecognizer | str = "tesseract",
    output: str = "text",
    recognizer_model: str | Path | None = None,
    model_cache_dir: str | Path | None = None,
    layout_processing: bool = True,
    device: str = "cpu",
    **kwargs,
):
    from newspaper_ocr.detectors import DETECTORS
    from newspaper_ocr.recognizers import RECOGNIZERS
    from newspaper_ocr.formatters import FORMATTERS
    from newspaper_ocr.layout_processor import LayoutProcessor

    if isinstance(detector, str):
        det_cls = DETECTORS.get(detector)
        detector = det_cls(model_dir=model_cache_dir)

    if isinstance(recognizer, str):
        rec_cls = RECOGNIZERS.get(recognizer)
        rec_kwargs = {}
        if recognizer_model:
            rec_kwargs["model"] = recognizer_model
        recognizer = rec_cls(**rec_kwargs)

    self.layout_processor = LayoutProcessor(enabled=layout_processing)

    if isinstance(output, str):
        fmt_cls = FORMATTERS.get(output)
        self.formatter = fmt_cls()
    else:
        self.formatter = output

    self.detector = detector
    self.recognizer = recognizer
```

- [ ] **Step 3: Update `__init__.py` exports**

```python
from newspaper_ocr.pipeline import Pipeline
from newspaper_ocr.models import BBox, Line, Region, PageLayout

__all__ = ["Pipeline", "BBox", "Line", "Region", "PageLayout"]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add string-based pipeline config with registry resolution"
```

---

### Task 8: CLI

**Files:**
- Create: `src/newspaper_ocr/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from newspaper_ocr.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "newspaper-ocr" in result.output.lower() or "backend" in result.output.lower()


def test_cli_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.jpg"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Implement CLI**

```python
# src/newspaper_ocr/cli.py
from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--backend", "-b", default="tesseract", help="Recognition backend (tesseract, effocr)")
@click.option("--output", "-o", default="text", help="Output format (text, json, hocr)")
@click.option("--model", "-m", default=None, help="Custom model path for recognizer")
@click.option("--model-dir", default=None, help="Model cache directory")
@click.option("--outdir", default=None, help="Output directory (default: stdout)")
def main(images, backend, output, model, model_dir, outdir):
    """OCR historical newspaper scans."""
    from newspaper_ocr import Pipeline

    pipe = Pipeline(
        recognizer=backend,
        output=output,
        recognizer_model=model,
        model_cache_dir=model_dir,
    )

    for image_path in images:
        result = pipe.ocr(image_path, output=output)

        if outdir:
            out_path = Path(outdir) / (Path(image_path).stem + _ext(output))
            Path(outdir).mkdir(parents=True, exist_ok=True)
            out_path.write_text(result)
            click.echo(f"Saved: {out_path}")
        else:
            click.echo(result)


def _ext(fmt: str) -> str:
    return {
        "text": ".txt",
        "json": ".json",
        "hocr": ".hocr",
    }.get(fmt, ".txt")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add CLI with click"
```

---

### Task 9: Integration Test and README

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/test_integration.py
"""End-to-end integration test using real models and a real newspaper scan."""
import pytest
import json
from pathlib import Path
from newspaper_ocr import Pipeline


@pytest.fixture
def real_image():
    path = Path("/Volumes/Lightning/chronicling-america/loc_downloads/sn84025908/1856-08-30/seq-4.jp2")
    if not path.exists():
        pytest.skip("Real newspaper scan not available")
    return str(path)


@pytest.fixture
def has_all_backends():
    try:
        pipe = Pipeline(recognizer="tesseract")
    except (ImportError, KeyError):
        pytest.skip("Required backends not available")


def test_full_pipeline_text(real_image, has_all_backends):
    pipe = Pipeline(recognizer="tesseract", output="text")
    result = pipe.ocr(real_image)
    assert len(result) > 100
    assert isinstance(result, str)


def test_full_pipeline_json(real_image, has_all_backends):
    pipe = Pipeline(recognizer="tesseract", output="json")
    result = pipe.ocr(real_image)
    data = json.loads(result)
    assert "regions" in data
    assert len(data["regions"]) > 0
    assert "lines" in data["regions"][0]
```

- [ ] **Step 2: Write README.md**

A concise README covering: installation, quick start (Python + CLI), backend options, output formats, and contributing.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add integration tests and README"
```

---

## Checkpoint Notes

**After Task 2:** The pipeline works end-to-end with mock backends. All subsequent tasks plug in real implementations without changing the pipeline code.

**After Task 5 (AS YOLO):** This is the hardest task — porting ~300 lines of detection code. The integration test with a real JP2 validates correctness. If the coordinate math is wrong, line crops will be visibly garbage.

**After Task 7:** `Pipeline(recognizer="tesseract")` works. This is the first "product-ready" state — you can use it to OCR newspapers.

**After Task 9:** The package is complete and ready for `pip install`.

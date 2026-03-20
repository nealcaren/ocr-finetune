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

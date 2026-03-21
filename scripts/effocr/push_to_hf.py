"""Push gold-standard OCR dataset to HuggingFace.

Uploads:
  - verified_lines.jsonl (13K+ line transcriptions with metadata)
  - sample_metadata.json (page sampling info)
  - {train,val,test}_images.tar.gz (line crop PNGs per split)
  - README.md (dataset card)

Usage:
    python scripts/effocr/push_to_hf.py [--repo NealCaren/newspaper-ocr-gold]
"""

import json
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"
VERIFIED_PATH = DATA_DIR / "100page_verified.jsonl"
SAMPLE_PATH = DATA_DIR / "100page_sample.json"
EXTRACTIONS_DIR = DATA_DIR / "100page_extractions"

DEFAULT_REPO = "NealCaren/newspaper-ocr-gold"


def count_lines_per_split(jsonl_path: Path) -> dict:
    counts = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            split = rec["split"]
            counts[split] = counts.get(split, 0) + 1
    return counts


def count_pages_per_split(extractions_dir: Path) -> dict:
    counts = {}
    for split in ("train", "val", "test"):
        split_dir = extractions_dir / split
        if split_dir.exists():
            counts[split] = len([d for d in split_dir.iterdir() if d.is_dir()])
        else:
            counts[split] = 0
    return counts


def get_flag_stats(jsonl_path: Path) -> dict:
    flags = {}
    for line in open(jsonl_path):
        rec = json.loads(line)
        flag = rec["flag"]
        flags[flag] = flags.get(flag, 0) + 1
    return flags


def create_dataset_card(repo_id: str, line_counts: dict, page_counts: dict) -> str:
    total_lines = sum(line_counts.values())
    total_pages = sum(page_counts.values())
    return f"""---
license: cc-by-4.0
task_categories:
  - image-to-text
tags:
  - ocr
  - historical-newspapers
  - fine-tuning
language:
  - en
size_categories:
  - 10K<n<100K
---

# newspaper-ocr-gold

Gold-standard OCR training data for historical newspaper scans.

## Contents
- {total_lines:,} line-level transcriptions verified by Qwen3-VL 235B
- Line crop PNG images from {total_pages} newspaper pages
- 73 unique titles spanning 1840s-2010s
- Train/val/test split by page (80/10/10)

## Splits
| Split | Pages | Lines |
|-------|-------|-------|
| train | {page_counts.get('train', 0)} | {line_counts.get('train', 0):,} |
| val | {page_counts.get('val', 0)} | {line_counts.get('val', 0):,} |
| test | {page_counts.get('test', 0)} | {line_counts.get('test', 0):,} |

## Files
- `verified_lines.jsonl` — full metadata (split, page_id, line_id, crop_path, transcription, confidence, flag)
- `sample_metadata.json` — page sampling details (73 titles, decade distribution)
- `train_images.tar.gz`, `val_images.tar.gz`, `test_images.tar.gz` — line crop PNGs organized as `{{split}}/{{page_id}}/lines/line_NNNN.png`

## Usage
```python
from huggingface_hub import hf_hub_download
import tarfile, json

# Download verified labels
path = hf_hub_download("NealCaren/newspaper-ocr-gold", "verified_lines.jsonl", repo_type="dataset")
with open(path) as f:
    lines = [json.loads(l) for l in f]

# Download and extract images for a split
for split in ["train", "val", "test"]:
    tar = hf_hub_download("NealCaren/newspaper-ocr-gold", f"{{split}}_images.tar.gz", repo_type="dataset")
    with tarfile.open(tar) as t:
        t.extractall("./gold_data/")
```

## Quality
- 49% clean, 47% partial (line cut-off at word boundary), 3% degraded
- Mean confidence: 0.95
- Verified by Qwen3-VL 235B via OpenRouter (blind transcription, no OCR input)
"""


def main():
    repo_id = DEFAULT_REPO
    if "--repo" in sys.argv:
        idx = sys.argv.index("--repo")
        repo_id = sys.argv[idx + 1]

    # Validate data exists
    for path, label in [
        (VERIFIED_PATH, "verified JSONL"),
        (EXTRACTIONS_DIR, "image extractions"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    # Gather stats
    line_counts = count_lines_per_split(VERIFIED_PATH)
    page_counts = count_pages_per_split(EXTRACTIONS_DIR)
    print(f"Lines per split: {line_counts}")
    print(f"Pages per split: {page_counts}")

    api = HfApi()

    # Create repo
    print(f"\nCreating/updating dataset repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload verified JSONL
    print("Uploading verified_lines.jsonl...")
    api.upload_file(
        path_or_fileobj=str(VERIFIED_PATH),
        path_in_repo="verified_lines.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Upload sample metadata
    if SAMPLE_PATH.exists():
        print("Uploading sample_metadata.json...")
        api.upload_file(
            path_or_fileobj=str(SAMPLE_PATH),
            path_in_repo="sample_metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Create and upload tar.gz per split
    for split in ("train", "val", "test"):
        split_dir = EXTRACTIONS_DIR / split
        if not split_dir.exists():
            print(f"WARNING: {split_dir} not found, skipping")
            continue

        tar_path = DATA_DIR / f"100page_{split}_images.tar.gz"
        print(f"Creating {tar_path.name}...")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(split_dir), arcname=split)

        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"  {tar_path.name}: {size_mb:.1f} MB")

        print(f"  Uploading {split}_images.tar.gz...")
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=f"{split}_images.tar.gz",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Clean up local tar
        tar_path.unlink()

    # Upload dataset card
    print("Uploading README.md...")
    card_content = create_dataset_card(repo_id, line_counts, page_counts)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(card_content)
        card_path = f.name
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    Path(card_path).unlink()

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nDone! Dataset published at: {url}")
    return url


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
sample_100_pages.py — Sample 100 diverse pages from LOC downloads.

Walks the LOC downloads directory, collects page metadata, and samples
100 pages spread across as many different titles (LCCNs) and decades as
possible. Splits by page into train/val/test (80/10/10).

Output: data/effocr/100page_sample.json
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

LOC_DIR = Path("/Volumes/Lightning/chronicling-america/loc_downloads")
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "effocr" / "100page_sample.json"
SEED = 42
N_PAGES = 100
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def collect_pages(loc_dir: Path) -> list[dict]:
    """Walk the LOC directory and collect all JP2 page metadata."""
    pages = []
    for lccn_dir in sorted(loc_dir.iterdir()):
        if not lccn_dir.is_dir():
            continue
        lccn = lccn_dir.name
        for date_dir in sorted(lccn_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            date_str = date_dir.name
            if not DATE_RE.match(date_str):
                continue
            year = int(date_str[:4])
            decade = (year // 10) * 10
            for jp2 in sorted(date_dir.glob("seq-*.jp2")):
                pages.append({
                    "path": str(jp2),
                    "lccn": lccn,
                    "date": date_str,
                    "year": year,
                    "decade": decade,
                })
    return pages


def sample_pages(pages: list[dict], n: int, seed: int) -> list[dict]:
    """
    Sample n pages with maximum diversity across LCCNs and decades.

    Strategy:
    1. Group pages by decade.
    2. Round-robin across decades (sorted), picking one page per round.
    3. Within each decade, prefer LCCNs not yet seen.
    """
    rng = random.Random(seed)

    by_decade: dict[int, list[dict]] = defaultdict(list)
    for p in pages:
        by_decade[p["decade"]].append(p)

    for decade in by_decade:
        rng.shuffle(by_decade[decade])

    decades = sorted(by_decade.keys())
    decade_queues = {d: list(by_decade[d]) for d in decades}

    selected = []
    seen_lccns: set[str] = set()
    decade_idx = 0

    while len(selected) < n:
        if decade_idx >= len(decades):
            decade_idx = 0

        decade = decades[decade_idx]
        queue = decade_queues[decade]

        if not queue:
            decades.pop(decade_idx)
            if not decades:
                break
            continue

        # Prefer an unseen LCCN
        chosen_idx = None
        for i, p in enumerate(queue):
            if p["lccn"] not in seen_lccns:
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = 0

        chosen = queue.pop(chosen_idx)
        selected.append(chosen)
        seen_lccns.add(chosen["lccn"])
        decade_idx += 1

    return selected


def split_pages(pages: list[dict], seed: int) -> dict:
    """Split pages into train/val/test (80/10/10) by page."""
    rng = random.Random(seed)
    shuffled = list(pages)
    rng.shuffle(shuffled)

    n_val = 10
    n_test = 10
    n_train = len(shuffled) - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def main() -> None:
    print(f"Walking {LOC_DIR} ...")
    pages = collect_pages(LOC_DIR)
    all_lccns = set(p["lccn"] for p in pages)
    all_decades = sorted(set(p["decade"] for p in pages))
    print(f"Found {len(pages):,} JP2 pages across {len(all_lccns)} LCCNs "
          f"and {len(all_decades)} decades.")

    sampled = sample_pages(pages, N_PAGES, SEED)
    splits = split_pages(sampled, SEED)

    # Report
    sampled_lccns = set(p["lccn"] for p in sampled)
    sampled_decades = sorted(set(p["decade"] for p in sampled))
    print(f"\nSampled {len(sampled)} pages from {len(sampled_lccns)} unique titles")
    print(f"Decades represented: {sampled_decades}")

    # Decade distribution
    decade_counts = defaultdict(int)
    for p in sampled:
        decade_counts[p["decade"]] += 1
    print("\nDecade distribution:")
    for d in sorted(decade_counts):
        print(f"  {d}s: {decade_counts[d]} pages")

    # Split info
    print(f"\nSplit sizes:")
    for split_name, split_pages_list in splits.items():
        lccns = set(p["lccn"] for p in split_pages_list)
        decades = sorted(set(p["decade"] for p in split_pages_list))
        print(f"  {split_name}: {len(split_pages_list)} pages, "
              f"{len(lccns)} titles, decades {decades[0]}s-{decades[-1]}s")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

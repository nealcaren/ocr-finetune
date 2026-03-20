"""
sample_pages.py — Sample 10 diverse pages from LOC downloads.

Walks the LOC downloads directory, collects page metadata, and samples
10 pages spread across different titles and decades using a round-robin
strategy across decades, preferring unseen LCCNs.

Output: data/effocr/pilot_pages.json
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow `from utils import ...` when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

LOC_DIR = Path("/Volumes/Lightning/chronicling-america/loc_downloads")
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "effocr" / "pilot_pages.json"
SEED = 42
N_PAGES = 10
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
                continue  # skip non-date dirs like ed-1/, gary-american-1928-current
            year = int(date_str[:4])
            decade = (year // 10) * 10
            for jp2 in sorted(date_dir.glob("seq-*.jp2")):
                pages.append(
                    {
                        "path": str(jp2),
                        "lccn": lccn,
                        "date": date_str,
                        "year": year,
                        "decade": decade,
                    }
                )
    return pages


def sample_pages(pages: list[dict], n: int, seed: int) -> list[dict]:
    """
    Sample n pages spread across different decades and LCCNs.

    Strategy:
    1. Group pages by decade.
    2. Round-robin across decades (sorted), picking one page per round.
    3. Within each decade, shuffle pages and prefer LCCNs not yet seen.
    """
    rng = random.Random(seed)

    # Group by decade
    by_decade: dict[int, list[dict]] = defaultdict(list)
    for p in pages:
        by_decade[p["decade"]].append(p)

    # Shuffle each decade's pages
    for decade in by_decade:
        rng.shuffle(by_decade[decade])

    decades = sorted(by_decade.keys())
    # Cyclic iterator over decades
    decade_queues = {d: list(by_decade[d]) for d in decades}

    selected = []
    seen_lccns: set[str] = set()
    decade_idx = 0

    while len(selected) < n:
        # If we've exhausted the round-robin, restart from beginning
        if decade_idx >= len(decades):
            decade_idx = 0

        decade = decades[decade_idx]
        queue = decade_queues[decade]

        if not queue:
            # This decade is exhausted; remove it and move on
            decades.pop(decade_idx)
            if not decades:
                break
            continue

        # Prefer an unseen LCCN; fall back to any available entry
        chosen_idx = None
        for i, p in enumerate(queue):
            if p["lccn"] not in seen_lccns:
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = 0  # fall back to first available

        chosen = queue.pop(chosen_idx)
        selected.append(chosen)
        seen_lccns.add(chosen["lccn"])
        decade_idx += 1

    return selected


def verify_paths(pages: list[dict]) -> None:
    missing = [p["path"] for p in pages if not Path(p["path"]).exists()]
    if missing:
        print(f"WARNING: {len(missing)} path(s) do not exist:")
        for m in missing:
            print(f"  {m}")
    else:
        print("All paths verified to exist.")


def main() -> None:
    print(f"Walking {LOC_DIR} ...")
    pages = collect_pages(LOC_DIR)
    print(f"Found {len(pages):,} JP2 pages across {len(set(p['lccn'] for p in pages))} LCCNs "
          f"and {len(set(p['decade'] for p in pages))} decades.")

    sampled = sample_pages(pages, N_PAGES, SEED)

    print(f"\nSelected {len(sampled)} pages:")
    for p in sampled:
        print(f"  lccn={p['lccn']}  date={p['date']}  decade={p['decade']}  {Path(p['path']).name}")

    verify_paths(sampled)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

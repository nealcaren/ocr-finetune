"""Generate benchmark.csv from benchmark-images directory."""
import csv
from pathlib import Path

images = sorted(
    p.name for p in Path("benchmark-images").iterdir()
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
)

with open("benchmark.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_name", "type", "source_dir"])
    for img in images:
        w.writerow([img, "newspaper", "benchmark-images"])

print(f"Created benchmark.csv with {len(images)} entries")

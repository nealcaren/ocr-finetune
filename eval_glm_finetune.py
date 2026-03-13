"""Run GLM-OCR (base + fine-tuned) on Inkbench images for CER/WER comparison.

Produces standard Inkbench accuracy tables plus a length-based breakdown
(short/medium/long reference texts) since GLM-OCR struggles with long texts.
"""

import csv
import subprocess
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

WORK = Path("/work/users/n/c/ncaren")
INKBENCH_DIR = WORK / "Inkbench"
BENCHMARK_CSV = INKBENCH_DIR / "benchmark.csv"
IMAGES_DIR = INKBENCH_DIR / "benchmark-images"
RESULTS_DIR = INKBENCH_DIR / "ocr-results"

MODELS = {
    "glm-ocr-base": "zai-org/GLM-OCR",
    "glm-ocr-finetuned": str(WORK / "glm-finetune/output/merged"),
}


def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return processor, model


def transcribe(processor, model, image_path):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Text Recognition:"},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)
    # Trim input tokens from output
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main():
    # Read benchmark CSV for image list
    with open(BENCHMARK_CSV) as f:
        reader = csv.DictReader(f)
        entries = [row for row in reader]
    print(f"Inkbench: {len(entries)} images")

    for model_name, model_path in MODELS.items():
        print(f"\nRunning {model_name} ({model_path})...")
        out_dir = RESULTS_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        processor, model = load_model(model_path)

        done = 0
        skipped = 0
        errors = 0
        for i, entry in enumerate(entries):
            stem = Path(entry["image_name"]).stem
            out_file = out_dir / f"{stem}.txt"
            if out_file.exists():
                skipped += 1
                continue  # resume support

            img_path = IMAGES_DIR / entry["image_name"]
            if not img_path.exists():
                print(f"  Missing image: {img_path}")
                errors += 1
                continue

            try:
                text = transcribe(processor, model, img_path)
                out_file.write_text(text, encoding="utf-8")
                done += 1
            except Exception as e:
                print(f"  Error on {entry['image_name']}: {e}")
                errors += 1
                continue

            if (done + skipped) % 50 == 0:
                print(f"  [{done + skipped}/{len(entries)}] "
                      f"{done} new, {skipped} cached, {errors} errors")

        print(f"  {model_name}: {done} transcribed, {skipped} cached, {errors} errors")
        print(f"  Results in {out_dir}")
        del model, processor
        torch.cuda.empty_cache()

    # Run Inkbench evaluation
    print("\n" + "=" * 60)
    print("Running evaluate_accuracy.py...")
    print("=" * 60)
    subprocess.run(
        ["python", str(INKBENCH_DIR / "evaluate_accuracy.py")],
        cwd=str(INKBENCH_DIR),
    )

    # Length-based breakdown
    print("\n" + "=" * 60)
    print("Accuracy by reference text length")
    print("=" * 60)
    length_breakdown()


def length_breakdown():
    """Break down accuracy by reference text length (short/medium/long)."""
    results_csv = INKBENCH_DIR / "ocr_eval_results.csv"
    ref_dir = INKBENCH_DIR / "benchmark-txt"

    if not results_csv.exists():
        print("(no results CSV found — skipping length breakdown)")
        return

    # Get word counts from reference texts
    ref_word_counts = {}
    for txt_file in ref_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        ref_word_counts[txt_file.stem] = len(text.split())

    # Length bins
    def length_bin(word_count):
        if word_count <= 75:
            return "Short (≤75 words)"
        elif word_count <= 200:
            return "Medium (76-200)"
        else:
            return "Long (200+)"

    # Aggregate CER by model × length bin
    # model -> bin -> list of cer_alnum values
    by_model_length = defaultdict(lambda: defaultdict(list))

    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok" or not row["cer_alnum"]:
                continue
            stem = Path(row["image_name"]).stem
            wc = ref_word_counts.get(stem, 0)
            if wc == 0:
                continue
            bin_name = length_bin(wc)
            cer = float(row["cer_alnum"])
            by_model_length[row["model"]][bin_name].append(cer)

    # Print table
    bins = ["Short (≤75 words)", "Medium (76-200)", "Long (200+)"]
    models = sorted(by_model_length.keys())

    # Header
    print(f"\n{'model':<25} ", end="")
    for b in bins:
        print(f"{b:<20} ", end="")
    print(f"{'Overall':<10}")
    print("-" * 95)

    for m in models:
        print(f"{m:<25} ", end="")
        all_vals = []
        for b in bins:
            vals = by_model_length[m][b]
            all_vals.extend(vals)
            if vals:
                acc = 1 - (sum(vals) / len(vals))
                print(f"{acc:.3f} (n={len(vals):<3})     ", end="")
            else:
                print(f"{'--':<20} ", end="")
        if all_vals:
            overall = 1 - (sum(all_vals) / len(all_vals))
            print(f"{overall:.3f}")
        else:
            print("--")

    # Also save as CSV
    out_csv = INKBENCH_DIR / "ocr_eval_by_length.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "length_bin", "n", "accuracy"])
        for m in models:
            for b in bins:
                vals = by_model_length[m][b]
                if vals:
                    acc = 1 - (sum(vals) / len(vals))
                    w.writerow([m, b, len(vals), round(acc, 4)])
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()

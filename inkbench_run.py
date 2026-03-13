"""Run models on Inkbench and evaluate accuracy.

Skips images that already have results (resume-safe). Use --force to redo.
Each model transcribes + evaluates independently, writing a per-model CSV.
Use --eval-only to aggregate all per-model CSVs into comparison tables.

Usage:
    # Run specific models (transcribe + per-model eval)
    python inkbench_run.py glm-ocr-base olmocr nanonets-ocr2

    # Run all known models
    python inkbench_run.py --all

    # Force re-run (overwrite existing results)
    python inkbench_run.py glm-ocr-base --force

    # Aggregate all per-model CSVs into comparison tables
    python inkbench_run.py --eval-only
"""

import argparse
import csv
import re
import signal
import time
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM,
    AutoModel, AutoTokenizer,
)

import jiwer
from jiwer import (
    Compose, ToLowerCase, RemoveMultipleSpaces, Strip,
    ReduceToListOfListOfWords, ReduceToListOfListOfChars,
)

OCR_TIMEOUT = 25  # seconds per image, same as production pipeline


class OCRTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise OCRTimeoutError("OCR generation timed out")


WORK = Path("/work/users/n/c/ncaren")
INKBENCH_DIR = WORK / "Inkbench"
BENCHMARK_CSV = INKBENCH_DIR / "benchmark.csv"
IMAGES_DIR = INKBENCH_DIR / "benchmark-images"
RESULTS_DIR = INKBENCH_DIR / "ocr-results"
REF_DIR = INKBENCH_DIR / "benchmark-txt"
EVAL_DIR = INKBENCH_DIR / "ocr-eval"  # per-model eval CSVs land here

# Model registry. Each entry maps a short name to its config.
#   "path":       HuggingFace model ID or local path
#   "loader":     which AutoModel class to use (default: AutoModelForImageTextToText)
#   "dtype":      torch dtype (default: bfloat16)
#   "trust_remote_code": passed to from_pretrained (default: False)
#   "prompt":     OCR prompt text sent as the user message (default: "Text Recognition:")

KNOWN_MODELS = {
    "glm-ocr-base": {
        "path": "zai-org/GLM-OCR",
    },
    "glm-ocr-finetuned": {
        "path": str(WORK / "glm-finetune/output/merged"),
    },
    "qwen3-vl-8b": {
        "path": "Qwen/Qwen3-VL-8B-Thinking",
        "trust_remote_code": True,
    },
    # --- Models from Multimodal-OCR3 Space ---
    "olmocr": {
        "path": "allenai/olmOCR-2-7B-1025",
        "trust_remote_code": True,
        "prompt": "Extract the plain text from this image.",
    },
    "nanonets-ocr2": {
        "path": "nanonets/Nanonets-OCR2-3B",
        "trust_remote_code": True,
        "prompt": "Extract the text from the above document as if you were reading it naturally.",
    },
    "chandra": {
        "path": "datalab-to/chandra",
        "trust_remote_code": True,
        "prompt": "Text Recognition:",
    },
    "dots-ocr": {
        "path": "prithivMLmods/Dots.OCR-Latest-BF16",
        "loader": "AutoModelForCausalLM",
        "trust_remote_code": True,
        "prompt": "Text Recognition:",
    },
    "deepseek-ocr2": {
        "path": "deepseek-ai/DeepSeek-OCR-2",
        "loader": "deepseek",
        "trust_remote_code": True,
        "prompt": "<image>\nFree OCR. ",
    },
    "rolmocr": {
        "path": "reducto/RolmOCR",
        "trust_remote_code": True,
        "prompt": "Extract the plain text from this image.",
    },
    # NOTE: MiniCPM-V 4.5 requires transformers<=4.57 (trust_remote_code model
    # uses _tied_weights_keys which was renamed in 5.x). Incompatible with our
    # env (>=5.1 for GLM-OCR). Would need a separate conda env to benchmark.
    # "minicpm-v-4.5": {
    #     "path": "openbmb/MiniCPM-V-4_5",
    #     "loader": "minicpm",
    #     "trust_remote_code": True,
    #     "prompt": "Text Recognition:",
    # },
}


# ── jiwer evaluation helpers ─────────────────────────────────────────

WORDS_TRANSFORM = Compose([
    ToLowerCase(), RemoveMultipleSpaces(), Strip(), ReduceToListOfListOfWords(),
])
CHARS_TRANSFORM = Compose([
    Strip(), ToLowerCase(), ReduceToListOfListOfChars(),
])
ALNUM_CHARS_ONLY = Compose([ReduceToListOfListOfChars()])
_ALNUM_RE = re.compile(r'[^0-9a-z]')


def _alnum_lower(s: str) -> str:
    return _ALNUM_RE.sub('', s.lower())


def compute_metrics(ref: str, hyp: str):
    """Return (wer, cer, cer_alnum) for a single reference/hypothesis pair."""
    w = jiwer.wer(ref, hyp,
                  reference_transform=WORDS_TRANSFORM,
                  hypothesis_transform=WORDS_TRANSFORM)
    c = jiwer.cer(ref, hyp,
                  reference_transform=CHARS_TRANSFORM,
                  hypothesis_transform=CHARS_TRANSFORM)
    ref_a, hyp_a = _alnum_lower(ref), _alnum_lower(hyp)
    c_alnum = jiwer.cer(ref_a, hyp_a,
                        reference_transform=ALNUM_CHARS_ONLY,
                        hypothesis_transform=ALNUM_CHARS_ONLY)
    return float(w), float(c), float(c_alnum)


def read_text(p: Path):
    if not p.exists():
        return None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return None


# ── Benchmark metadata ───────────────────────────────────────────────

def load_benchmark():
    """Return list of {image_name, type, stem} from benchmark.csv."""
    entries = []
    with open(BENCHMARK_CSV) as f:
        for row in csv.DictReader(f):
            img = row["image_name"].strip()
            entries.append({
                "image_name": img,
                "type": row.get("type", "").strip(),
                "stem": Path(img).stem,
            })
    return entries


# ── Model loading & transcription ────────────────────────────────────

def _get_model_config(model_name):
    """Return full config dict for a model name, with defaults filled in."""
    if model_name in KNOWN_MODELS:
        cfg = dict(KNOWN_MODELS[model_name])
    else:
        cfg = {"path": model_name}
    cfg.setdefault("loader", "AutoModelForImageTextToText")
    cfg.setdefault("dtype", "bfloat16")
    cfg.setdefault("trust_remote_code", False)
    cfg.setdefault("prompt", "Text Recognition:")
    return cfg


def load_model(cfg):
    """Load processor/tokenizer + model according to config."""
    path = cfg["path"]
    dtype = getattr(torch, cfg["dtype"])
    trust = cfg["trust_remote_code"]
    loader = cfg["loader"]

    if loader == "deepseek":
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, use_safetensors=True,
            torch_dtype=dtype,
        ).eval().cuda()
        return tokenizer, model

    if loader == "minicpm":
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=dtype,
        ).eval().cuda()
        return tokenizer, model

    # Standard path: AutoProcessor + AutoModel*
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust)

    if loader == "AutoModelForCausalLM":
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map="cuda", trust_remote_code=trust,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype, device_map="cuda", trust_remote_code=trust,
        )
    model.eval()
    return processor, model


def transcribe(processor_or_tokenizer, model, image_path, prompt="Text Recognition:",
               loader="AutoModelForImageTextToText"):
    """Transcribe an image. Dispatches to model-specific inference as needed."""
    image = Image.open(image_path).convert("RGB")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(OCR_TIMEOUT)
    try:
        if loader == "deepseek":
            return _transcribe_deepseek(processor_or_tokenizer, model, image_path, prompt)
        if loader == "minicpm":
            return _transcribe_minicpm(processor_or_tokenizer, model, image, prompt)
        return _transcribe_standard(processor_or_tokenizer, model, image, prompt)
    finally:
        signal.alarm(0)


def _transcribe_standard(processor, model, image, prompt):
    """Standard processor + generate() path (GLM-OCR, olmOCR, Nanonets, Chandra, Dots, RolmOCR)."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Some models (e.g. Dots.OCR) add keys the model doesn't accept
    inputs.pop("mm_token_type_ids", None)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _transcribe_deepseek(tokenizer, model, image_path, prompt):
    """DeepSeek-OCR-2 uses model.infer() with its own image loading."""
    res = model.infer(
        tokenizer, prompt=prompt, image_file=str(image_path),
        base_size=1024, image_size=768, crop_mode=True, save_results=False,
    )
    # infer() returns a string or list; extract text
    if isinstance(res, list):
        return "\n".join(str(r) for r in res).strip()
    return str(res).strip()


def _transcribe_minicpm(tokenizer, model, image, prompt):
    """MiniCPM-V uses model.chat() with a custom message format."""
    msgs = [{"role": "user", "content": [image, prompt]}]
    answer = model.chat(msgs=msgs, tokenizer=tokenizer,
                        enable_thinking=False, stream=False)
    return str(answer).strip()


# ── Per-model: transcribe + evaluate ─────────────────────────────────

def evaluate_model(model_name, entries):
    """Compute CER/WER for a single model against reference texts.

    Reads transcription results from RESULTS_DIR / model_name.
    Writes per-image metrics to EVAL_DIR / {model_name}.csv.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EVAL_DIR / f"{model_name}.csv"
    hyp_dir = RESULTS_DIR / model_name

    fieldnames = ["image_name", "type", "model", "status",
                  "wer", "cer", "cer_alnum"]
    rows = []
    ok = 0

    for entry in entries:
        stem = entry["stem"]
        ref_text = read_text(REF_DIR / f"{stem}.txt")
        hyp_text = read_text(hyp_dir / f"{stem}.txt")

        if ref_text is None:
            rows.append({"image_name": entry["image_name"], "type": entry["type"],
                         "model": model_name, "status": "missing_ref",
                         "wer": "", "cer": "", "cer_alnum": ""})
            continue
        if hyp_text is None:
            rows.append({"image_name": entry["image_name"], "type": entry["type"],
                         "model": model_name, "status": "missing_hyp",
                         "wer": "", "cer": "", "cer_alnum": ""})
            continue

        w, c, c_alnum = compute_metrics(ref_text, hyp_text)
        rows.append({"image_name": entry["image_name"], "type": entry["type"],
                      "model": model_name, "status": "ok",
                      "wer": w, "cer": c, "cer_alnum": c_alnum})
        ok += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print quick summary
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        mean_cer = sum(float(r["cer_alnum"]) for r in ok_rows) / len(ok_rows)
        accuracy = 1 - mean_cer
        print(f"  {model_name}: accuracy={accuracy:.3f} "
              f"(CER_alnum={mean_cer:.4f}, n={len(ok_rows)})")
    else:
        print(f"  {model_name}: no valid results to evaluate")

    print(f"  Saved per-image metrics to {out_csv}")


def run_model(model_name, image_files, entries, force=False):
    """Transcribe all benchmark images, then evaluate."""
    cfg = _get_model_config(model_name)

    print(f"\n{'=' * 60}")
    print(f"Running {model_name} ({cfg['path']})")
    print(f"{'=' * 60}")

    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    processor_or_tokenizer, model = load_model(cfg)
    prompt = cfg["prompt"]
    loader = cfg["loader"]

    done = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for img_path in image_files:
        stem = img_path.stem
        out_file = out_dir / f"{stem}.txt"

        if out_file.exists() and not force:
            skipped += 1
            continue

        try:
            text = transcribe(processor_or_tokenizer, model, img_path,
                              prompt=prompt, loader=loader)
            out_file.write_text(text, encoding="utf-8")
            done += 1
        except OCRTimeoutError:
            print(f"  Timeout on {img_path.name}")
            out_file.write_text("[TIMEOUT]", encoding="utf-8")
            errors += 1
            continue
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
            errors += 1
            continue

        if (done + skipped) % 50 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = len(image_files) - (done + skipped + errors)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{done + skipped}/{len(image_files)}] "
                  f"{done} new, {skipped} cached, {errors} errors | "
                  f"{rate:.1f} img/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  {model_name}: {done} transcribed, {skipped} cached, {errors} errors "
          f"({elapsed:.0f}s)")

    del model, processor_or_tokenizer
    torch.cuda.empty_cache()

    # Evaluate this model's results
    print(f"\n  Evaluating {model_name}...")
    evaluate_model(model_name, entries)


# ── Aggregation: combine per-model CSVs ──────────────────────────────

TYPE_LABELS = {
    "BOOK_PAGE": "Book Page",
    "HANDWRITTEN": "Handwritten",
    "MIXED": "Mixed",
    "OTHER_TYPED_OR_PRINTED": "Other Typed/Printed",
    # Also accept labels already mapped (e.g. from benchmark.csv with raw labels)
    "newspaper": "Newspaper",
}
TYPE_ORDER = ["Book Page", "Handwritten", "Mixed", "Other Typed/Printed", "Newspaper"]


def aggregate_results():
    """Generate missing per-model eval CSVs, then produce comparison tables."""
    # First, evaluate any models that have results but no eval CSV yet
    entries = load_benchmark()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_DIR.exists():
        for model_dir in sorted(RESULTS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            eval_csv = EVAL_DIR / f"{model_dir.name}.csv"
            if not eval_csv.exists():
                txt_count = len(list(model_dir.glob("*.txt")))
                if txt_count > 0:
                    print(f"Generating eval for {model_dir.name} ({txt_count} results)...")
                    evaluate_model(model_dir.name, entries)

    model_csvs = sorted(EVAL_DIR.glob("*.csv"))
    if not model_csvs:
        print("No results to evaluate. Run models first.")
        return

    # Collect all rows
    all_rows = []
    for csv_path in model_csvs:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)

    # Aggregate by model and by model × type
    by_model = defaultdict(list)           # model -> [cer_alnum values]
    by_model_type = defaultdict(lambda: defaultdict(list))  # model -> type -> [cer_alnum]

    for row in all_rows:
        if row["status"] != "ok" or not row["cer_alnum"]:
            continue
        m = row["model"]
        cer = float(row["cer_alnum"])
        by_model[m].append(cer)
        typ = TYPE_LABELS.get(row["type"], row["type"])
        by_model_type[m][typ].append(cer)

    models = sorted(by_model.keys())

    # ── Summary table to stdout ──
    print(f"\n{'=' * 60}")
    print("INKBENCH RESULTS (accuracy = 1 − mean CER_alnum)")
    print(f"{'=' * 60}")

    # Determine which type columns actually have data
    active_types = [t for t in TYPE_ORDER if any(by_model_type[m].get(t) for m in models)]

    header = f"{'model':<25} {'Overall':<12} "
    header += " ".join(f"{t:<20}" for t in active_types)
    print(f"\n{header}")
    print("-" * len(header))

    for m in sorted(models, key=lambda m: sum(by_model[m]) / len(by_model[m])):
        vals = by_model[m]
        overall = 1 - sum(vals) / len(vals)
        line = f"{m:<25} {overall:<12.3f} "
        for t in active_types:
            tv = by_model_type[m].get(t, [])
            if tv:
                acc = 1 - sum(tv) / len(tv)
                line += f"{acc:.3f} (n={len(tv):<3})     "
            else:
                line += f"{'--':<20} "
        print(line)

    # ── Write combined CSV ──
    combined_csv = INKBENCH_DIR / "ocr_eval_results.csv"
    fieldnames = ["image_name", "type", "model", "status", "wer", "cer", "cer_alnum"]
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCombined results: {combined_csv}")

    # ── Write accuracy wide CSV ──
    accuracy_csv = INKBENCH_DIR / "ocr_eval_model_accuracy.csv"
    acc_headers = ["model", "Overall"] + active_types
    acc_rows = []
    for m in models:
        vals = by_model[m]
        row = {"model": m, "Overall": round(1 - sum(vals) / len(vals), 4)}
        for t in active_types:
            tv = by_model_type[m].get(t, [])
            if tv:
                row[t] = round(1 - sum(tv) / len(tv), 4)
            else:
                row[t] = ""
        acc_rows.append(row)
    acc_rows.sort(key=lambda r: -r["Overall"])

    with open(accuracy_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=acc_headers)
        writer.writeheader()
        writer.writerows(acc_rows)
    print(f"Accuracy table:   {accuracy_csv}")

    # ── Length breakdown ──
    ref_word_counts = {}
    for txt_file in REF_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        ref_word_counts[txt_file.stem] = len(text.split())

    def length_bin(wc):
        if wc <= 75:
            return "Short (≤75 words)"
        elif wc <= 200:
            return "Medium (76-200)"
        return "Long (200+)"

    by_model_length = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        if row["status"] != "ok" or not row["cer_alnum"]:
            continue
        stem = Path(row["image_name"]).stem
        wc = ref_word_counts.get(stem, 0)
        if wc == 0:
            continue
        by_model_length[row["model"]][length_bin(wc)].append(float(row["cer_alnum"]))

    bins = ["Short (≤75 words)", "Medium (76-200)", "Long (200+)"]
    print(f"\n{'model':<25} ", end="")
    for b in bins:
        print(f"{b:<20} ", end="")
    print(f"{'Overall':<10}")
    print("-" * 95)

    for m in sorted(models, key=lambda m: sum(by_model[m]) / len(by_model[m])):
        print(f"{m:<25} ", end="")
        all_vals = []
        for b in bins:
            vals = by_model_length[m].get(b, [])
            all_vals.extend(vals)
            if vals:
                acc = 1 - sum(vals) / len(vals)
                print(f"{acc:.3f} (n={len(vals):<3})     ", end="")
            else:
                print(f"{'--':<20} ", end="")
        if all_vals:
            print(f"{1 - sum(all_vals)/len(all_vals):.3f}")
        else:
            print("--")

    length_csv = INKBENCH_DIR / "ocr_eval_by_length.csv"
    with open(length_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "length_bin", "n", "accuracy"])
        for m in models:
            for b in bins:
                vals = by_model_length[m].get(b, [])
                if vals:
                    w.writerow([m, b, len(vals), round(1 - sum(vals) / len(vals), 4)])
    print(f"Length breakdown:  {length_csv}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run models on Inkbench benchmark")
    parser.add_argument("models", nargs="*",
                        help=f"Model names to run. Known: {', '.join(KNOWN_MODELS.keys())}")
    parser.add_argument("--all", action="store_true", help="Run all known models")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe even if results exist")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip transcription, aggregate existing per-model CSVs")
    parser.add_argument("-n", "--num-images", type=int, default=0,
                        help="Only process first N images (0 = all)")
    args = parser.parse_args()

    if args.eval_only:
        aggregate_results()
        return

    if not args.models and not args.all:
        parser.error("Specify model names or use --all")

    entries = load_benchmark()

    # Find benchmark images
    image_files = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    if args.num_images > 0:
        image_files = image_files[:args.num_images]
    print(f"Inkbench: {len(image_files)} images")

    # Determine which models to run
    if args.all:
        model_names = list(KNOWN_MODELS.keys())
    else:
        model_names = []
        for name in args.models:
            if name in KNOWN_MODELS:
                model_names.append(name)
            else:
                # Treat as a HuggingFace model path — register it on the fly
                short_name = name.replace("/", "-").lower()
                KNOWN_MODELS[short_name] = {"path": name, "trust_remote_code": True}
                model_names.append(short_name)

    for model_name in model_names:
        run_model(model_name, image_files, entries, force=args.force)


if __name__ == "__main__":
    main()

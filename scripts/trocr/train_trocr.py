#!/usr/bin/env python3
"""
train_trocr.py — Fine-tune TrOCR-base-printed on historical newspaper lines.

Usage:
    # Local test
    python scripts/trocr/train_trocr.py --local --epochs 2 --max-train 100

    # Longleaf
    python scripts/trocr/train_trocr.py --epochs 10 --batch-size 16
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "microsoft/trocr-base-printed"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "effocr"


class OCRDataset(Dataset):
    def __init__(self, manifest_path, processor, max_target_length=128, limit=0):
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []

        manifest_dir = manifest_path.parent
        for line in open(manifest_path):
            p = line.strip()
            resolved = str((manifest_dir / p).resolve())
            gt_path = resolved.replace(".png", ".gt.txt")
            if Path(gt_path).exists() and Path(gt_path).stat().st_size > 0:
                self.samples.append((resolved, gt_path))

        if limit > 0:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        text = open(gt_path).read().strip()

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length, truncation=True
        ).input_ids
        # Replace padding token id with -100 so it's ignored in loss
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def safe_decode(processor, ids):
    """Decode token IDs, clipping out-of-range values."""
    import numpy as np
    vocab_size = processor.tokenizer.vocab_size
    ids = np.clip(ids, 0, vocab_size - 1)
    try:
        return processor.batch_decode(ids, skip_special_tokens=True)
    except (OverflowError, ValueError):
        # Fallback: decode one at a time, skip failures
        results = []
        for row in ids:
            try:
                results.append(processor.tokenizer.decode(row, skip_special_tokens=True))
            except:
                results.append("")
        return results


def compute_metrics(pred, processor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = safe_decode(processor, pred_ids)
    label_str = safe_decode(processor, labels_ids)

    # CER
    total_chars = 0
    total_errors = 0
    for pred_s, label_s in zip(pred_str, label_str):
        ref, hyp = label_s.strip(), pred_s.strip()
        if not ref:
            continue
        m, n = len(ref), len(hyp)
        d = list(range(n + 1))
        for i in range(1, m + 1):
            prev = d[:]
            d[0] = i
            for j in range(1, n + 1):
                c = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + c)
        total_errors += d[n]
        total_chars += m

    cer = total_errors / total_chars if total_chars > 0 else 1.0
    return {"cer": cer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    if args.local:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if args.batch_size > 4:
            args.batch_size = 2
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent.parent / "data" / "trocr" / "output"
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Find training data — try multiple locations
    train_manifest = None
    val_manifest = None

    if data_dir and (data_dir / "train_manifest.txt").exists():
        train_manifest = data_dir / "train_manifest.txt"
        val_manifest = data_dir / "val_manifest.txt"
    else:
        # Try kraken manifests (local dev)
        kraken_dir = Path(__file__).resolve().parent.parent / "kraken"
        for name in ["train_multires_manifest.txt", "train_full_manifest.txt"]:
            if (kraken_dir / name).exists():
                train_manifest = kraken_dir / name
                val_manifest = kraken_dir / name.replace("train_", "val_")
                break

    if not train_manifest or not train_manifest.exists():
        # Last resort: download from HF and create local data
        print("No local manifests found. Downloading from HF dataset...")
        from datasets import load_dataset
        ds = load_dataset("NealCaren/newspaper-ocr-gold")
        hf_dir = output_dir.parent / "hf_data"
        hf_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val"]:
            split_data = ds[split if split in ds else "test"]
            r2 = split_data.filter(lambda x: x["resolution"] == "r2")
            manifest_lines = []
            for i, row in enumerate(r2):
                img_path = hf_dir / f"{split}_{i:05d}.png"
                gt_path = hf_dir / f"{split}_{i:05d}.gt.txt"
                if not img_path.exists():
                    row["image"].save(str(img_path))
                    gt_path.write_text(row["transcription"])
                manifest_lines.append(str(img_path))
            manifest_path = hf_dir / f"{split}_manifest.txt"
            manifest_path.write_text("\n".join(manifest_lines) + "\n")
            print(f"  {split}: {len(manifest_lines)} images")
        train_manifest = hf_dir / "train_manifest.txt"
        val_manifest = hf_dir / "val_manifest.txt"

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_dir}")

    print("\nLoading processor and model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Set decoder config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    print("Loading datasets...")
    train_ds = OCRDataset(train_manifest, processor, limit=args.max_train)
    val_ds = OCRDataset(val_manifest, processor, limit=args.max_eval)
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=128,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=device == "cuda",
        fp16=False,
        eval_strategy="no" if args.local else "epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False if args.local else True,
        metric_for_best_model="cer" if not args.local else None,
        greater_is_better=False if not args.local else None,
        logging_steps=50,
        dataloader_num_workers=0 if args.local else 4,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda p: compute_metrics(p, processor),
    )

    print("\nStarting training...")
    trainer.train()

    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nSaved to {final_dir}")


if __name__ == "__main__":
    main()

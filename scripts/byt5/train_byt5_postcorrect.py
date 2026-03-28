#!/usr/bin/env python3
"""
train_byt5_postcorrect.py — Fine-tune ByT5-small for OCR post-correction.

Takes blocks of noisy Tesseract lines and learns to output clean, joined text
with dehyphenation and error correction.

Usage:
    # Local (MPS/CPU):
    python scripts/byt5/train_byt5_postcorrect.py --local --epochs 10

    # Longleaf (CUDA):
    python scripts/byt5/train_byt5_postcorrect.py --epochs 10 --batch-size 8
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "byt5"
MODEL_NAME = "google/byt5-small"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 1024


class PostCorrectionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_length, max_target_length):
        with open(data_path) as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = self.tokenizer(
            ex["input"],
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        targets = self.tokenizer(
            ex["target"],
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs


def safe_decode(tokenizer, ids):
    """Decode token IDs, filtering out-of-range values (ByT5 generation artifact)."""
    # ByT5 offset is 3 (0=pad, 1=eos, 2=unk), valid byte range is 3..258
    special = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    filtered = [i for i in ids if i not in special and 3 <= i <= 258]
    try:
        return tokenizer.decode(filtered, skip_special_tokens=True)
    except (ValueError, OverflowError):
        # Last resort: manually convert valid byte tokens
        return "".join(chr(i - 3) for i in filtered if 3 <= i <= 258)


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    # Replace -100 with pad token
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label]
              for label in labels]

    decoded_preds = [safe_decode(tokenizer, p) for p in preds]
    decoded_labels = [safe_decode(tokenizer, l) for l in labels]

    # CER
    total_chars = 0
    total_errors = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        ref = label.strip()
        hyp = pred.strip()
        if not ref:
            continue
        # Levenshtein
        m, n = len(ref), len(hyp)
        d = list(range(n + 1))
        for i in range(1, m + 1):
            prev = d[:]
            d[0] = i
            for j in range(1, n + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
        total_errors += d[n]
        total_chars += m

    cer = total_errors / total_chars if total_chars > 0 else 1.0
    return {"cer": cer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--local", action="store_true",
                        help="Run on MPS/CPU (smaller batch)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: data/byt5/)")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Limit training examples (0 = all)")
    parser.add_argument("--max-eval", type=int, default=0,
                        help="Limit eval examples (0 = all)")
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

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "output"

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {output_dir}")

    # Load tokenizer and model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Load datasets — prefer pipeline versions
    train_path = data_dir / "train_pipeline.json"
    val_path = data_dir / "val_pipeline.json"
    if not train_path.exists():
        train_path = data_dir / "train_gold.json"
        val_path = data_dir / "val_gold.json"
    if not train_path.exists():
        train_path = data_dir / "train.json"
        val_path = data_dir / "val.json"

    if not train_path.exists():
        print(f"ERROR: No training data found. Run build_gold_regions.py first.")
        return

    print("Loading datasets...")
    train_ds = PostCorrectionDataset(train_path, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_ds = PostCorrectionDataset(val_path, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH) if val_path.exists() else None

    if args.max_train > 0:
        train_ds.examples = train_ds.examples[:args.max_train]
    if args.max_eval > 0 and val_ds:
        val_ds.examples = val_ds.examples[:args.max_eval]

    print(f"  Train: {len(train_ds)} examples")
    if val_ds:
        print(f"  Val: {len(val_ds)} examples")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2 if not args.local else 4,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=device == "cuda",
        fp16=False,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=1000 if not args.local else 700,
        save_strategy="steps",
        save_steps=1000 if not args.local else 700,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="cer" if val_ds else None,
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=1,
        dataloader_num_workers=0 if args.local else 4,
        remove_unused_columns=False,
        report_to="none",
    )

    # Data collator for variable-length padding
    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    # Resume from checkpoint if available
    last_checkpoint = None
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"\nResuming from {last_checkpoint}")

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved final model to {final_dir}")


if __name__ == "__main__":
    main()

"""Fine-tune GLM-OCR with LoRA using HuggingFace Trainer + PEFT directly.

Skips LLaMA-Factory to avoid transformers version conflicts
(GLM-OCR requires transformers >= 5.1.0).

Usage:
    python train_glm_ocr.py
    python train_glm_ocr.py --epochs 5 --batch-size 1 --lr 5e-5
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
)

WORK = Path("/work/users/n/c/ncaren")
DATA_DIR = WORK / "glm-finetune"
OUTPUT_DIR = DATA_DIR / "output" / "lora"


class OCRDataset(TorchDataset):
    """Lazily loads and tokenizes images on the fly to avoid OOM."""

    def __init__(self, json_path: Path, processor):
        with open(json_path) as f:
            self.records = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = rec["images"][0]
        image = Image.open(image_path).convert("RGB")
        ground_truth = rec["messages"][1]["content"]

        # Build the prompt the same way GLM-OCR expects
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Text Recognition:"},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize prompt + ground truth together
        full_text = prompt + ground_truth + self.processor.tokenizer.eos_token
        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)

        # Create labels: mask the prompt tokens with -100, only train on the response
        prompt_inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
        }


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    max_len = max(x["input_ids"].shape[0] for x in batch)
    pad_token_id = 0  # GLM uses 0 as pad

    input_ids = []
    attention_mask = []
    labels = []
    pixel_values = []
    image_grid_thw = []

    for x in batch:
        seq_len = x["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([
            x["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=x["input_ids"].dtype),
        ]))
        attention_mask.append(torch.cat([
            x["attention_mask"],
            torch.zeros(pad_len, dtype=x["attention_mask"].dtype),
        ]))
        labels.append(torch.cat([
            x["labels"],
            torch.full((pad_len,), -100, dtype=x["labels"].dtype),
        ]))
        pixel_values.append(x["pixel_values"])
        image_grid_thw.append(x["image_grid_thw"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "pixel_values": torch.stack(pixel_values),
        "image_grid_thw": torch.stack(image_grid_thw),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
    model = AutoModelForImageTextToText.from_pretrained(
        "zai-org/GLM-OCR",
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets lazily
    print("Loading training data...")
    train_file = DATA_DIR / "train.json"
    val_file = DATA_DIR / "val.json"

    train_dataset = OCRDataset(train_file, processor)
    print(f"Train: {len(train_dataset)} examples")

    val_dataset = None
    if val_file.exists():
        val_dataset = OCRDataset(val_file, processor)
        print(f"Val: {len(val_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        per_device_eval_batch_size=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    print(f"Saving adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()

"""Merge LoRA adapter back into base GLM-OCR model.

Usage:
    python merge_lora_model.py
    python merge_lora_model.py --adapter-dir /path/to/lora --output-dir /path/to/merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText

WORK = Path("/work/users/n/c/ncaren")
DEFAULT_ADAPTER = WORK / "glm-finetune/output/lora"
DEFAULT_OUTPUT = WORK / "glm-finetune/output/merged"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        "zai-org/GLM-OCR", torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")

    print(f"Loading adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))

    print("Merging weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    print("Done!")


if __name__ == "__main__":
    main()

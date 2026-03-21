#!/usr/bin/env python3
"""
Fine-tune EfficientOCR char recognizer on downscaled newspaper character crops.

EffOCR's training pipeline expects:
1. An ImageFolder structure: <ready_to_go_dir>/<class_id>/{PAIRED_*.png, synth_*.png}
2. PAIRED_* files = real crop data (our downscaled chars)
3. Non-PAIRED files = synthetic font renders (used as KNN reference set)

This script:
1. Generates synthetic font renders for each character class in our training data
2. Runs EffOCR's built-in training (contrastive metric learning + KNN eval)

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/finetune_effocr.py [--epochs 3] [--batch-size 64]
"""

import sys
import os
import json
import argparse
import multiprocessing
from pathlib import Path

# Must set start method before any torch imports on macOS
# to avoid pickle errors with lambda transforms in DataLoader workers
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
from PIL import Image, ImageFont, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "effocr"))

DATA_DIR = PROJECT_ROOT / "data" / "effocr"
CHAR_DATA_DIR = DATA_DIR / "training_data" / "char"
WORD_DATA_DIR = DATA_DIR / "training_data" / "word"
FINETUNED_DIR = DATA_DIR / "finetuned_models"


def generate_synth_renders(data_dir: Path, font_size: int = 32, num_fonts: int = 1):
    """
    Generate synthetic font renders for each character class.

    Creates non-PAIRED images that EffOCR uses as the KNN reference set.
    Uses system fonts available on macOS.
    """
    # Find available fonts
    font_paths = []
    font_search_dirs = [
        "/System/Library/Fonts",
        "/Library/Fonts",
    ]
    for fdir in font_search_dirs:
        fdir_path = Path(fdir)
        if fdir_path.exists():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                font_paths.extend(fdir_path.glob(ext))

    if not font_paths:
        print("WARNING: No system fonts found, using PIL default font")
        font_paths = [None]
    else:
        # Pick a few good fonts
        preferred = ["Helvetica.ttc", "Times New Roman.ttf", "Courier New.ttf",
                     "Arial.ttf", "Georgia.ttf"]
        selected = []
        for pref in preferred:
            matches = [f for f in font_paths if f.name == pref]
            if matches:
                selected.append(matches[0])
        if not selected:
            selected = font_paths[:5]
        font_paths = selected[:num_fonts]

    print(f"Using {len(font_paths)} font(s) for synth renders")

    # Get all character class directories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    total_renders = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        # Parse ASCII code(s) from folder name
        try:
            char = chr(int(class_name))
        except (ValueError, OverflowError):
            # Multi-char or invalid — skip
            continue

        # Check if synth renders already exist
        existing_synth = [f for f in class_dir.iterdir()
                         if f.is_file() and not f.name.startswith("PAIRED")]
        if existing_synth:
            total_renders += len(existing_synth)
            continue

        # Generate renders
        for font_idx, font_path in enumerate(font_paths):
            try:
                if font_path is None:
                    font = ImageFont.load_default()
                else:
                    font = ImageFont.truetype(str(font_path), size=font_size)
            except Exception:
                font = ImageFont.load_default()

            # Render the character
            # Create a temporary image to measure text size
            tmp = Image.new("RGB", (100, 100), (255, 255, 255))
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), char, font=font)
            tw = bbox[2] - bbox[0] + 8  # padding
            th = bbox[3] - bbox[1] + 8
            tw = max(tw, 16)
            th = max(th, 16)

            img = Image.new("RGB", (tw, th), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Center the character
            x = (tw - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (th - (bbox[3] - bbox[1])) // 2 - bbox[1]
            draw.text((x, y), char, fill=(0, 0, 0), font=font)

            font_name = font_path.stem if font_path else "default"
            # Use 0x<hex> naming convention (EffOCR legacy format)
            # save_ref_index handles "0x" prefix: chr(int(basename.split("_")[0], base=16))
            hex_code = f"0x{ord(char):04X}"
            render_name = f"{hex_code}_{font_name}.png"
            # Only save one render per class with this name
            render_path = class_dir / render_name
            if not render_path.exists():
                img.save(str(render_path))
                total_renders += 1

    print(f"Generated/found {total_renders} synth renders across {len(class_dirs)} classes")
    return total_renders


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EffOCR char recognizer")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate (default: 0.002)")
    parser.add_argument("--target", choices=["char", "word", "both"], default="char",
                        help="Which recognizer to fine-tune (default: char)")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Skip synthetic render generation")
    args = parser.parse_args()

    targets = ["char", "word"] if args.target == "both" else [args.target]

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Fine-tuning {target} recognizer")
        print(f"{'='*60}")

        if target == "char":
            data_dir = CHAR_DATA_DIR
        else:
            data_dir = WORD_DATA_DIR

        if not data_dir.exists():
            print(f"ERROR: Training data not found: {data_dir}")
            print("Run build_training_data.py first.")
            sys.exit(1)

        # Count data
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        paired_count = sum(
            1 for d in class_dirs
            for f in d.iterdir()
            if f.is_file() and f.name.startswith("PAIRED")
        )
        print(f"Training data: {paired_count} PAIRED crops in {len(class_dirs)} classes")

        if paired_count == 0:
            print("ERROR: No training data found!")
            sys.exit(1)

        # Generate synth renders
        if not args.skip_synth:
            print("\nGenerating synthetic font renders...")
            generate_synth_renders(data_dir)

        # Set up output directory
        output_dir = FINETUNED_DIR / f"{target}_recognizer"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Initialize EffOCR with timm backend for training
        print("\nInitializing EffOCR for training...")
        from efficient_ocr import EffOCR

        hf_repo = "dell-research-harvard/effocr_en"
        config = {
            "Global": {
                "skip_line_detection": True,
                "wandb_project": None,
            },
            "Recognizer": {
                "char": {
                    "model_backend": "timm" if target == "char" else "onnx",
                    "model_dir": str(output_dir) if target == "char" else str(DATA_DIR / "models"),
                    # hf_repo_id must be None for timm backend (no .pth on HF)
                    # This loads fresh ImageNet-pretrained mobilenetv3 weights
                    "hf_repo_id": None if target == "char" else f"{hf_repo}/char_recognizer",
                    "device": "cpu",
                    "training": {
                        "ready_to_go_data_dir_path": str(CHAR_DATA_DIR),
                        "batch_size": args.batch_size,
                        "num_epochs": args.epochs,
                        "lr": args.lr,
                        "m": 4,
                        "finetune": True,
                        "test_at_end": False,
                        "hardneg_k": None,
                        "num_passes": 1,
                        "start_epoch": 1,
                        "train_val_test_split": [0.7, 0.15, 0.15],
                        "render_dict": None,
                        "font_dir_path": None,
                        "hns_txt_path": None,
                        "train_set_from_coco_json": None,
                        "val_set_from_coco_json": None,
                        "test_set_from_coco_json": None,
                        "char_trans_version": 2,
                        "dec_lr_factor": 0.9,
                        "expansion_factor": 1,
                        "imsize": 224,
                        "adamw_beta1": 0.9,
                        "adamw_beta2": 0.999,
                        "temp": 0.1,
                        "weight_decay": 0.0005,
                        "epoch_viz_dir": None,
                        "few_shot": None,
                        "int_eval_steps": None,
                        "ascender": True,
                        "aug_paired": False,
                        "char_only_sampler": False,
                        "diff_sizes": False,
                        "high_blur": False,
                        "latin_suggested_augs": True,
                        "lr_schedule": False,
                        "no_aug": False,
                        "pretrain": False,
                        "default_font_name": "Noto",
                    },
                },
                "word": {
                    "model_backend": "timm" if target == "word" else "onnx",
                    "model_dir": str(output_dir) if target == "word" else str(DATA_DIR / "models"),
                    "hf_repo_id": None if target == "word" else f"{hf_repo}/word_recognizer",
                    "device": "cpu",
                    "training": {
                        "ready_to_go_data_dir_path": str(WORD_DATA_DIR),
                        "batch_size": args.batch_size,
                        "num_epochs": args.epochs,
                        "lr": args.lr,
                        "m": 4,
                        "finetune": True,
                        "test_at_end": False,
                        "hardneg_k": None,
                        "num_passes": 1,
                        "start_epoch": 1,
                        "train_val_test_split": [0.7, 0.15, 0.15],
                        "render_dict": None,
                        "font_dir_path": None,
                        "hns_txt_path": None,
                        "train_set_from_coco_json": None,
                        "val_set_from_coco_json": None,
                        "test_set_from_coco_json": None,
                        "char_trans_version": 2,
                        "dec_lr_factor": 0.9,
                        "expansion_factor": 1,
                        "imsize": 224,
                        "adamw_beta1": 0.9,
                        "adamw_beta2": 0.999,
                        "temp": 0.1,
                        "weight_decay": 0.0005,
                        "epoch_viz_dir": None,
                        "few_shot": None,
                        "int_eval_steps": None,
                        "ascender": True,
                        "aug_paired": False,
                        "char_only_sampler": False,
                        "diff_sizes": False,
                        "high_blur": False,
                        "latin_suggested_augs": True,
                        "lr_schedule": False,
                        "no_aug": False,
                        "pretrain": False,
                        "default_font_name": "Noto",
                    },
                },
            },
            "Localizer": {
                "model_backend": "onnx",
                "model_dir": str(DATA_DIR / "models"),
                "hf_repo_id": hf_repo,
            },
            "Line": {
                "model_backend": "onnx",
                "model_dir": str(DATA_DIR / "models"),
                "hf_repo_id": hf_repo,
            },
        }

        effocr = EffOCR(config=config)
        print("EffOCR initialized.")

        # Run training
        print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
        print("(This may be slow on CPU — that's expected)")

        try:
            effocr.train(target=f"{target}_recognizer")
            print(f"\n{target} recognizer training complete!")
            print(f"Model saved to: {output_dir}")
        except Exception as e:
            print(f"\nERROR during training: {e}")
            import traceback
            traceback.print_exc()
            print("\nTraining failed — see error above.")
            sys.exit(1)


if __name__ == "__main__":
    main()

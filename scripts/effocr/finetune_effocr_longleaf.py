#!/usr/bin/env python3
"""
Fine-tune EfficientOCR char/word recognizer on Longleaf HPC (L40S GPU).

This is the GPU-accelerated version of finetune_effocr.py, designed for
UNC's Longleaf cluster. Training data is built from the gold-standard
HuggingFace dataset (NealCaren/newspaper-ocr-gold) by setup_longleaf.sh.

Usage:
    python scripts/effocr/finetune_effocr_longleaf.py --target char --epochs 50 --batch-size 256 --device cuda
    python scripts/effocr/finetune_effocr_longleaf.py --target word --epochs 50 --batch-size 128 --device cuda
"""

import sys
import os
import argparse
import multiprocessing
from pathlib import Path

# Must set start method before any torch imports
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

WORK = Path("/work/users/n/c/ncaren")
EFFOCR_DIR = WORK / "effocr-finetune"
TRAINING_DATA_DIR = EFFOCR_DIR / "training_data"
OUTPUT_DIR = EFFOCR_DIR / "output"


def generate_synth_renders(data_dir: Path, font_size: int = 32):
    """
    Generate synthetic font renders for each character class.

    Creates non-PAIRED images that EffOCR uses as the KNN reference set.
    Uses fonts available on the system (Longleaf has basic fonts in /usr/share/fonts).
    """
    from PIL import Image, ImageFont, ImageDraw

    font_paths = []
    font_search_dirs = [
        "/usr/share/fonts/truetype",
        "/usr/share/fonts",
        "/usr/share/fonts/dejavu",
    ]
    for fdir in font_search_dirs:
        fdir_path = Path(fdir)
        if fdir_path.exists():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                font_paths.extend(fdir_path.rglob(ext))

    if not font_paths:
        print("WARNING: No system fonts found, using PIL default font")
        font_paths = [None]
    else:
        # Pick up to 2 fonts
        font_paths = font_paths[:2]

    print(f"Using {len(font_paths)} font(s) for synth renders")

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    total_renders = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        try:
            char = chr(int(class_name))
        except (ValueError, OverflowError):
            continue

        # Check if synth renders already exist
        existing_synth = [f for f in class_dir.iterdir()
                         if f.is_file() and not f.name.startswith("PAIRED")]
        if existing_synth:
            total_renders += len(existing_synth)
            continue

        for font_idx, font_path in enumerate(font_paths):
            try:
                if font_path is None:
                    font = ImageFont.load_default()
                else:
                    font = ImageFont.truetype(str(font_path), size=font_size)
            except Exception:
                font = ImageFont.load_default()

            tmp = Image.new("RGB", (100, 100), (255, 255, 255))
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), char, font=font)
            tw = max(bbox[2] - bbox[0] + 8, 16)
            th = max(bbox[3] - bbox[1] + 8, 16)

            img = Image.new("RGB", (tw, th), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            x = (tw - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (th - (bbox[3] - bbox[1])) // 2 - bbox[1]
            draw.text((x, y), char, fill=(0, 0, 0), font=font)

            font_name = font_path.stem if font_path else "default"
            hex_code = f"0x{ord(char):04X}"
            render_name = f"{hex_code}_{font_name}.png"
            render_path = class_dir / render_name
            if not render_path.exists():
                img.save(str(render_path))
                total_renders += 1

    print(f"Generated/found {total_renders} synth renders across {len(class_dirs)} classes")
    return total_renders


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EffOCR on Longleaf (GPU)")
    parser.add_argument("--target", choices=["char", "word"], required=True,
                        help="Which recognizer to fine-tune")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate (default: 0.002)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Skip synthetic render generation")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"EffOCR Fine-tuning on Longleaf")
    print(f"  Target: {args.target} recognizer")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}")

    data_dir = TRAINING_DATA_DIR / args.target
    output_dir = OUTPUT_DIR / f"{args.target}_recognizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: Training data not found: {data_dir}")
        print(f"Run setup_longleaf.sh to download gold data from HuggingFace and build training data.")
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
        print("ERROR: No PAIRED training crops found!")
        sys.exit(1)

    # Generate synth renders (needed for KNN reference set)
    if not args.skip_synth:
        print("\nGenerating synthetic font renders...")
        generate_synth_renders(data_dir)

    # Build config -- uses nested 'training' dict matching EffOCR's default_config.py
    print("\nInitializing EffOCR for training...")
    from efficient_ocr import EffOCR

    config = {
        "Global": {
            "single_model_training": f"{args.target}_recognizer",
            "wandb_project": None,
        },
        "Recognizer": {
            args.target: {
                "model_backend": "timm",
                "model_dir": str(output_dir),
                "hf_repo_id": None,  # None for timm = fresh ImageNet-pretrained mobilenetv3
                "device": args.device,
                "training": {
                    "ready_to_go_data_dir_path": str(data_dir),
                    "batch_size": args.batch_size,
                    "num_epochs": args.epochs,
                    "lr": args.lr,
                    "m": 4,
                    "finetune": True,
                    "test_at_end": True,
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
    }

    effocr = EffOCR(config=config)
    print("EffOCR initialized.")

    # Run training
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    try:
        effocr.train(target=f"{args.target}_recognizer")
        print(f"\n{args.target} recognizer training complete!")
        print(f"Model saved to: {output_dir}")
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

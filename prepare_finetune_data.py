"""Convert NealCaren/OCRTrain to ShareGPT format for GLM-OCR fine-tuning."""

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("/work/users/n/c/ncaren/glm-finetune")
IMAGES_DIR = OUTPUT_DIR / "images"
TRAIN_FILE = OUTPUT_DIR / "train.json"
VAL_FILE = OUTPUT_DIR / "val.json"


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("NealCaren/OCRTrain", split="train")
    print(f"Loaded {len(ds)} examples")

    # 90/10 train/val split
    split = ds.train_test_split(test_size=0.1, seed=42)

    for split_name, subset in [("train", split["train"]), ("val", split["test"])]:
        records = []
        for i, example in enumerate(subset):
            # Save image
            img_name = f"{split_name}_{i:04d}.png"
            img_path = IMAGES_DIR / img_name
            example["image"].save(img_path)

            records.append({
                "messages": [
                    {"role": "user", "content": "<image>Text Recognition:"},
                    {"role": "assistant", "content": example["text"]},
                ],
                "images": [str(img_path)],
            })

        out_file = TRAIN_FILE if split_name == "train" else VAL_FILE
        out_file.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"{split_name}: {len(records)} examples → {out_file}")

    # Write dataset_info.json for LLaMA-Factory
    dataset_info = {
        "newspaper_ocr": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
        }
    }
    info_file = OUTPUT_DIR / "dataset_info.json"
    info_file.write_text(json.dumps(dataset_info, indent=2))
    print(f"Dataset info → {info_file}")


if __name__ == "__main__":
    main()

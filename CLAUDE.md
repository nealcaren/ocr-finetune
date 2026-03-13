# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoRA fine-tuning pipeline for `zai-org/GLM-OCR` on historical document images, targeting improved OCR accuracy on degraded early 20th-century newspaper print. Runs on UNC's Longleaf HPC cluster (L40S GPU, 48GB VRAM). Evaluated against the Inkbench benchmark (400 images, CER/WER metrics).

## Execution Pipeline

All scripts run on Longleaf. The repo is cloned at `/work/users/n/c/ncaren/ocr-finetune/` and data lives at `/work/users/n/c/ncaren/glm-finetune/`.

```
1. bash setup_finetune.sh          # login node: create conda env, install deps, prepare data (~15 min)
2. sbatch run_finetune.sl          # compute node: LoRA training + merge adapter into base model (~1-2 hrs)
3. sbatch run_eval.sl              # compute node: run base + finetuned on Inkbench (~2 hrs)
4. watch -n 30 ./finetune_dashboard.sh   # monitor progress
```

## Architecture

**Data prep** (`prepare_finetune_data.py`): Downloads `NealCaren/OCRTrain` from HuggingFace, saves images to disk, creates ShareGPT-format `train.json` (90%) and `val.json` (10%).

**Training** (`train_glm_ocr.py`): Custom HF `Trainer` + `peft` LoRA (not LLaMA-Factory — it caps at transformers ≤4.57.1, but GLM-OCR requires ≥5.1.0). Uses a custom `OCRDataset` that lazily tokenizes images and masks prompt tokens in labels (`-100`) so loss is only on the assistant response. Custom `collate_fn` handles variable-length padding.

**Merge** (`merge_lora_model.py`): Merges LoRA adapter into base model for inference. Output at `/work/.../glm-finetune/output/merged`.

**Evaluation** (`inkbench_run.py`): Multi-model benchmark runner with resume support, 25s per-image timeout, and length-based accuracy breakdown (short/medium/long). Calls Inkbench's `evaluate_accuracy.py` for CER/WER tables. `eval_glm_finetune.py` is an older single-purpose version.

**SLURM jobs**: `run_finetune.sl` (L40S, 64GB RAM, 6hr) chains train→merge. `run_eval.sl` (L40S, 48GB RAM, 4hr) runs evaluation.

## Key Constraints

- All caches (`HF_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`, `XDG_CACHE_HOME`, `TMPDIR`) must point to `/work/` — the home directory has a 50GB quota.
- `PYTHONNOUSERSITE=1` is set in SLURM scripts to avoid `~/.local` package conflicts.
- GLM-OCR requires `transformers>=5.1`. After pip installing, torch must be force-reinstalled with `--no-deps` to fix CUDA driver issues.
- The conda environment lives at `/work/users/n/c/ncaren/envs/glm-finetune`.
- Hardcoded paths throughout use `/work/users/n/c/ncaren/` as the base.

## Training Data Format

ShareGPT JSON with image paths and `<image>Text Recognition:` as the user prompt:

```json
[{"messages": [{"role": "user", "content": "<image>Text Recognition:"},
               {"role": "assistant", "content": "transcribed text"}],
  "images": ["/work/.../images/train_0001.png"]}]
```

## Known Models (inkbench_run.py registry)

- `glm-ocr-base`: `zai-org/GLM-OCR`
- `glm-ocr-finetuned`: `/work/.../glm-finetune/output/merged`
- `qwen3-vl-8b`: `Qwen/Qwen3-VL-8B-Thinking`

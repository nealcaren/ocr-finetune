# Fine-Tuning GLM-OCR on Longleaf

## Goal

Fine-tune `zai-org/GLM-OCR` on historical document images to improve OCR accuracy on degraded early 20th-century print. Start with the `NealCaren/OCRTrain` dataset (1,200 image+text pairs), then swap in gold-standard newspaper corrections once available. Evaluate using the [Inkbench](https://github.com/nealcaren/Inkbench) framework (CER/WER against reference transcriptions).

## Strategy

**LoRA fine-tuning** on a single L40S GPU (48GB VRAM), using HuggingFace `Trainer` + `peft` directly (no LLaMA-Factory — it caps at transformers ≤4.57.1, but GLM-OCR requires ≥5.1.0).

Why LoRA:

- Faster training (~1 hour for 1,200 examples × 3 epochs)
- Smaller checkpoint (~50MB adapter vs ~4GB full model)
- Easy to A/B test: load base model + adapter, compare against base model
- Can always upgrade to full SFT later if LoRA results are promising

The merged model stays on Longleaf at `/work/users/n/c/ncaren/glm-finetune/output/merged` — no HuggingFace upload needed. Our OCR pipeline already runs on Longleaf and can point directly at this path.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_finetune.sh` | Create conda env, install deps, prepare data (login node) |
| `prepare_finetune_data.py` | Convert NealCaren/OCRTrain → ShareGPT JSON format |
| `train_glm_ocr.py` | LoRA training with HF Trainer + PEFT |
| `merge_lora_model.py` | Merge LoRA adapter into base model |
| `run_finetune.sl` | SLURM job: train + merge |
| `eval_glm_finetune.py` | Run base + fine-tuned on Inkbench, compute CER/WER |
| `run_eval.sl` | SLURM job: evaluation |
| `finetune_dashboard.sh` | Live monitoring: `watch -n 30 ./finetune_dashboard.sh` |

## Data Format

Training data is ShareGPT JSON consumed by our custom training script:

```json
[
  {
    "messages": [
      {"role": "user", "content": "<image>Text Recognition:"},
      {"role": "assistant", "content": "The recognized text goes here"}
    ],
    "images": ["images/example_001.png"]
  }
]
```

`prepare_finetune_data.py` downloads OCRTrain from HuggingFace, saves images to disk, creates `train.json` (90%) and `val.json` (10%), and writes `dataset_info.json`.

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Base model | `zai-org/GLM-OCR` |
| Method | LoRA (rank=8, alpha=16, all-linear) |
| Epochs | 3 |
| Batch size | 2 (× 4 grad accumulation = effective 8) |
| Learning rate | 1e-4, cosine decay, 10% warmup |
| Precision | bf16 |
| Logging | Every 10 steps |
| Checkpoints | Every 100 steps, keep last 3 |

## Environment

All caches redirected to `/work` to avoid 50GB home quota:

| Variable | Path |
|----------|------|
| `CONDA_PKGS_DIRS` | `/work/.../conda_pkgs` |
| `PIP_CACHE_DIR` | `/work/.../pip_cache` |
| `XDG_CACHE_HOME` | `/work/.../.cache` |
| `HF_HOME` | `/work/.../hf_cache` |
| `TMPDIR` | `/work/.../tmp` |

Key deps: `transformers>=5.1`, `peft`, `accelerate`, `datasets`, `jiwer`

## Evaluation: Inkbench

400 historical document images with Library of Congress volunteer transcriptions. Computes CER and WER broken down by document type (Book Page, Handwritten, Mixed, Other Typed/Printed). Accuracy = 1 − CER.

`eval_glm_finetune.py` runs both base and fine-tuned GLM-OCR on all 400 images, then calls `evaluate_accuracy.py`.

## Execution Order

1. `bash setup_finetune.sh` — create env, install deps, prepare data (login node, ~15 min)
2. `sbatch run_finetune.sl` — train LoRA adapter + merge (compute node, ~1–2 hours)
3. `sbatch run_eval.sl` — run base + fine-tuned on 400 Inkbench images, compute CER/WER (~2 hours)
4. Review `ocr_eval_model_accuracy.csv` — if improved, update `GLM_OCR_MODEL` path in `ocr_newspapers.py`

## Using the Fine-Tuned Model

```python
# In ocr_newspapers.py, change:
GLM_OCR_MODEL = "zai-org/GLM-OCR"
# To:
GLM_OCR_MODEL = "/work/users/n/c/ncaren/glm-finetune/output/merged"
```

## Future: Gold-Standard Newspaper Data

Once we have manually corrected newspaper OCR (from the review website), replace the training data:

1. Export corrections from the review site as image+text pairs
2. Re-run `prepare_finetune_data.py` with the new dataset
3. Retrain — the pipeline is identical, just swap the data source

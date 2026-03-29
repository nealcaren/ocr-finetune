#!/bin/bash
#SBATCH -J effocr_finetune
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH -t 6:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o effocr_finetune_%j.out
#SBATCH -e effocr_finetune_%j.err

WORK=/work/users/n/c/ncaren
EFFOCR_DIR=$WORK/effocr-finetune
GOLD_DIR=$EFFOCR_DIR/gold_data
VENV=$WORK/envs/effocr-uv

# Activate uv venv
source "$VENV/bin/activate"

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

set -e

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify gold data exists (should be downloaded by setup_longleaf.sh)
if [ ! -f "$GOLD_DIR/verified_lines.jsonl" ]; then
    echo "ERROR: Gold data not found at $GOLD_DIR/"
    echo "Run setup_longleaf.sh first to download from HuggingFace."
    exit 1
fi

# Build EffOCR training data from gold labels (if not already built)
TRAIN_CHAR_DIR=$EFFOCR_DIR/training_data/char
PAIRED_COUNT=$(find $TRAIN_CHAR_DIR -name "PAIRED_*.png" 2>/dev/null | head -1 | wc -l)
if [ "$PAIRED_COUNT" -gt 0 ]; then
    CHAR_CLASSES=$(find $TRAIN_CHAR_DIR -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "Training data already built: $CHAR_CLASSES char classes, skipping build step"
else
    echo "=== Building EffOCR training data (fast, localizer-only) ==="
    python $WORK/ocr-finetune/scripts/effocr/build_training_data_fast.py \
        --gold-jsonl $GOLD_DIR/verified_lines.jsonl \
        --image-dir $GOLD_DIR \
        --output-dir $EFFOCR_DIR/training_data \
        --model-dir $EFFOCR_DIR/models \
        --scales 1.0,0.5,0.35,0.25
fi

# Train char recognizer (50 epochs, large batch size for GPU)
echo ""
echo "=== Training char recognizer ==="
python $WORK/ocr-finetune/scripts/effocr/finetune_effocr_longleaf.py \
    --target char --epochs 50 --batch-size 256 --device cuda

# Train word recognizer (50 epochs, smaller batch for longer sequences)
echo ""
echo "=== Training word recognizer ==="
python $WORK/ocr-finetune/scripts/effocr/finetune_effocr_longleaf.py \
    --target word --epochs 50 --batch-size 128 --device cuda

echo ""
echo "Done at $(date)"

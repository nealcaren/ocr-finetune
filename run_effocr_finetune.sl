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

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/effocr

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export PIP_CACHE_DIR=$WORK/pip_cache
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1  # ignore ~/.local packages

set -e  # stop on first error

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
if [ -d "$TRAIN_CHAR_DIR" ] && [ "$(ls -A $TRAIN_CHAR_DIR 2>/dev/null)" ]; then
    echo "Training data already built, skipping build step"
else
    echo "=== Building EffOCR training data from gold labels ==="
    python $WORK/ocr-finetune/scripts/effocr/build_training_data.py \
        --gold-jsonl $GOLD_DIR/verified_lines.jsonl \
        --image-dir $GOLD_DIR \
        --output-dir $EFFOCR_DIR/training_data \
        --resolutions 32 64 128
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

#!/bin/bash
#SBATCH -J byt5_postcorrect
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -t 6:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o byt5_finetune_%j.out
#SBATCH -e byt5_finetune_%j.err

WORK=/work/users/n/c/ncaren
REPO=$WORK/ocr-finetune

# Use the uv venv
source $WORK/envs/effocr-uv/bin/activate

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1

set -e

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install any missing deps
uv pip install accelerate datasets 2>/dev/null || true

# Copy training data from repo to work dir
echo "=== Setting up training data ==="
mkdir -p $WORK/byt5-finetune/data
cp $REPO/data/byt5/train_pipeline.json $WORK/byt5-finetune/data/train_pipeline.json
cp $REPO/data/byt5/val_pipeline.json $WORK/byt5-finetune/data/val_pipeline.json
cp $REPO/data/byt5/test_pipeline.json $WORK/byt5-finetune/data/test_pipeline.json 2>/dev/null || true
echo "Train: $(python -c "import json; print(len(json.load(open('$WORK/byt5-finetune/data/train_pipeline.json'))))")"
echo "Val: $(python -c "import json; print(len(json.load(open('$WORK/byt5-finetune/data/val_pipeline.json'))))")"

# Train
echo ""
echo "=== Training ByT5-small (20 epochs) ==="
python $REPO/scripts/byt5/train_byt5_postcorrect.py \
    --epochs 20 \
    --batch-size 8 \
    --data-dir $WORK/byt5-finetune/data \
    --output-dir $WORK/byt5-finetune/output

echo ""
echo "Done at $(date)"

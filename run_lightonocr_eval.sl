#!/bin/bash
#SBATCH -J lightonocr_eval
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 2:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o lightonocr_eval_%j.out
#SBATCH -e lightonocr_eval_%j.err

WORK=/work/users/n/c/ncaren
REPO=$WORK/ocr-finetune

source $WORK/envs/effocr-uv/bin/activate

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

set -e

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install LightOnOCR deps
uv pip install "git+https://github.com/huggingface/transformers" pypdfium2 2>/dev/null || true

# Eval baseline on test regions
echo "=== Evaluating LightOnOCR-2-1B-base on R2 regions ==="
python $REPO/scripts/lightonocr/eval_lightonocr.py --limit 200

echo ""
echo "Done at $(date)"

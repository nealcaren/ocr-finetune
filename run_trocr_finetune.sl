#!/bin/bash
#SBATCH -J trocr_finetune
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -t 3:00:00
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o trocr_finetune_%j.out
#SBATCH -e trocr_finetune_%j.err

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

echo "=== Training TrOCR-base-printed (10 epochs, multi-res) ==="
python $REPO/scripts/trocr/train_trocr.py \
    --epochs 10 \
    --batch-size 16 \
    --output-dir $WORK/trocr-finetune/output

echo ""
echo "Done at $(date)"

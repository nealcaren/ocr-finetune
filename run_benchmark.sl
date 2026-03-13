#!/bin/bash
#SBATCH -J ocr_bench_%a
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -t 4:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_%A_%a.out
#SBATCH -e benchmark_%A_%a.err
#SBATCH --array=0-4

# Each array task gets its own GPU and runs one model: transcribe + evaluate.
#
# Usage:
#   sbatch run_benchmark.sl                        # all 8 models, full benchmark
#   sbatch --array=0,2 run_benchmark.sl            # olmocr + chandra only
#   NUM_IMAGES=10 sbatch run_benchmark.sl          # proof-of-concept with 10 images
#
# After all jobs finish, aggregate into comparison tables:
#   python inkbench_run.py --eval-only

#   0=olmocr  1=chandra  2=dots-ocr  3=rolmocr  4=glm-ocr-base
MODELS=(olmocr chandra dots-ocr rolmocr glm-ocr-base)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Default to all images; override with NUM_IMAGES env var
NUM_IMAGES=${NUM_IMAGES:-0}

WORK=/work/users/n/c/ncaren

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/glm-finetune

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1

set -e

echo "Job $SLURM_JOB_ID (array $SLURM_ARRAY_TASK_ID) — model: $MODEL"
echo "Host: $(hostname) at $(date)"
echo "NUM_IMAGES: $NUM_IMAGES (0 = all)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install extra deps needed by various models (idempotent)
pip install -q qwen-vl-utils sentencepiece addict easydict einops 2>/dev/null || true

# Transcribe + compute per-model CER/WER. Resume-safe.
python $WORK/ocr-finetune/inkbench_run.py "$MODEL" -n "$NUM_IMAGES"

echo "$MODEL done at $(date)"

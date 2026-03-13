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
#SBATCH --array=0-3

# Each array task gets its own GPU and runs one model: transcribe + evaluate.
# All 4 run in parallel. Per-model eval CSVs land in ocr-eval/.
#
# Usage:
#   sbatch run_benchmark.sl                        # all 4 models in parallel
#   sbatch --array=0,2 run_benchmark.sl            # olmocr + chandra only
#
# After all jobs finish, aggregate into comparison tables:
#   python inkbench_run.py --eval-only

MODELS=(olmocr nanonets-ocr2 chandra dots-ocr)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install extra deps needed by some models (idempotent)
pip install -q qwen-vl-utils sentencepiece 2>/dev/null || true

# Transcribe 400 images + compute per-model CER/WER. Resume-safe.
python $WORK/ocr-finetune/inkbench_run.py "$MODEL"

echo "$MODEL done at $(date)"

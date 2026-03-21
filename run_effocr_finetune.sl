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

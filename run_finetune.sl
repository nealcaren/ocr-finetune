#!/bin/bash
#SBATCH -J glm_finetune
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH -t 6:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o finetune_%j.out
#SBATCH -e finetune_%j.err

WORK=/work/users/n/c/ncaren

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/glm-finetune

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export PIP_CACHE_DIR=$WORK/pip_cache
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1  # ignore ~/.local packages

set -e  # stop on first error

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Train LoRA adapter
python $WORK/ocr-finetune/train_glm_ocr.py

# Merge into full model (only runs if training succeeded)
python $WORK/ocr-finetune/merge_lora_model.py

echo "Done at $(date)"

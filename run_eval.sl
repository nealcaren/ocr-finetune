#!/bin/bash
#SBATCH -J glm_eval
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH -t 4:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o eval_%j.out
#SBATCH -e eval_%j.err

WORK=/work/users/n/c/ncaren

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/glm-finetune

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run both models on Inkbench (skips existing results by default)
python $WORK/ocr-finetune/inkbench_run.py glm-ocr-base glm-ocr-finetuned

echo "Done at $(date)"

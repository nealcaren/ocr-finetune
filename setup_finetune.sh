#!/bin/bash
# Run this interactively on the Longleaf login node to create the fine-tuning environment.
# Usage: bash setup_finetune.sh

set -e

WORK=/work/users/n/c/ncaren

# Redirect ALL caches to /work to avoid filling 50GB home directory quota
export PIP_CACHE_DIR=$WORK/pip_cache
export TMPDIR=$WORK/tmp
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1  # ignore ~/.local packages
mkdir -p $PIP_CACHE_DIR $TMPDIR $CONDA_PKGS_DIRS $XDG_CACHE_HOME

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"

# New env for fine-tuning (on /work, not home)
conda create --yes --prefix $WORK/envs/glm-finetune python=3.11
conda activate $WORK/envs/glm-finetune

# PyTorch with CUDA
conda install --yes -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1

# Transformers 5.1+ (required for GLM-OCR support) + PEFT for LoRA
pip install "transformers>=5.1" peft accelerate

# Dataset loading + evaluation deps (install with all sub-deps)
pip install datasets pillow jiwer

# Fix torch — pip packages overwrite CUDA libs causing cudaGetDriverEntryPointByVersion error
# Use --no-deps so it doesn't downgrade transformers back below 5.1
pip install --force-reinstall --no-deps torch torchvision

# Verify the full import chain works
python -c "from datasets import Dataset; from peft import LoraConfig; from transformers import AutoProcessor; print('All imports OK')"

# Pre-download GLM-OCR to /work cache (may fail on login node due to no GPU — that's OK)
export HF_HOME=$WORK/hf_cache
python -c "
from transformers import AutoProcessor
AutoProcessor.from_pretrained('zai-org/GLM-OCR')
print('GLM-OCR processor cached.')
print('Note: model weights will download on first GPU job if not already cached.')
" || echo "Processor download failed (may need internet) — will retry on compute node."

# Prepare training data
mkdir -p $WORK/glm-finetune
cd $WORK/ocr-finetune
python prepare_finetune_data.py

# Clone Inkbench for evaluation
if [ ! -d "$WORK/Inkbench" ]; then
    cd $WORK
    git clone https://github.com/nealcaren/Inkbench.git
fi

# Fix hardcoded Mac path in evaluate_accuracy.py
sed -i 's|/Users/nealcaren/Downloads/InkBench/benchmark.csv|benchmark.csv|' $WORK/Inkbench/evaluate_accuracy.py

# Generate benchmark.csv from image files
cd $WORK/Inkbench
python $WORK/ocr-finetune/make_benchmark_csv.py

echo ""
echo "Setup complete."
echo "  Environment: $WORK/envs/glm-finetune"
echo "  Training data: $WORK/glm-finetune/"
echo "  Inkbench: $WORK/Inkbench/"
echo ""
echo "Next steps:"
echo "  1. Submit training: sbatch run_finetune.sl"
echo "  2. After training: sbatch run_eval.sl"

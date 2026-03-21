#!/bin/bash
# Set up EffOCR fine-tuning environment on Longleaf login node.
# Usage: bash scripts/effocr/setup_longleaf.sh

set -e

WORK=/work/users/n/c/ncaren
EFFOCR_DIR=$WORK/effocr-finetune

# Redirect ALL caches to /work to avoid filling 50GB home directory quota
export PIP_CACHE_DIR=$WORK/pip_cache
export TMPDIR=$WORK/tmp
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONNOUSERSITE=1  # ignore ~/.local packages
mkdir -p $PIP_CACHE_DIR $TMPDIR $CONDA_PKGS_DIRS $XDG_CACHE_HOME

# Create directory structure
mkdir -p $EFFOCR_DIR/{training_data/char,training_data/word,output,models}

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"

# Create dedicated env for EffOCR (separate from glm-finetune due to different deps)
if [ -d "$WORK/envs/effocr" ]; then
    echo "Conda env already exists: $WORK/envs/effocr"
    conda activate $WORK/envs/effocr
else
    echo "Creating conda env: $WORK/envs/effocr"
    conda create --yes --prefix $WORK/envs/effocr python=3.11
    conda activate $WORK/envs/effocr

    # PyTorch with CUDA
    conda install --yes -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1

    # EffOCR from fork (includes training pipeline fixes)
    pip install git+https://github.com/nealcaren/efficient_ocr.git

    # Additional deps for EffOCR training
    pip install timm faiss-cpu pytorch-metric-learning

    # Fix torch -- pip packages may overwrite CUDA libs
    pip install --force-reinstall --no-deps torch torchvision

    # Verify imports
    python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
from efficient_ocr import EffOCR
print('EffOCR import OK')
import timm
print(f'timm {timm.__version__}')
"
fi

echo ""
echo "Setup complete."
echo "  Environment: $WORK/envs/effocr"
echo "  Training data dir: $EFFOCR_DIR/training_data/"
echo "  Output dir: $EFFOCR_DIR/output/"
echo ""
echo "Next steps:"
echo "  1. Rsync training data from local machine:"
echo "     rsync -av data/effocr/training_data/ longleaf:$EFFOCR_DIR/training_data/"
echo "  2. Submit training job:"
echo "     sbatch run_effocr_finetune.sl"

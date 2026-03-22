#!/bin/bash
# Set up EffOCR fine-tuning environment on Longleaf login node.
# Downloads gold-standard data from HuggingFace, builds EffOCR training data.
# Usage: bash scripts/effocr/setup_longleaf.sh

set -e

WORK=/work/users/n/c/ncaren
EFFOCR_DIR=$WORK/effocr-finetune

# Redirect ALL caches to /work to avoid filling 50GB home directory quota
export PIP_CACHE_DIR=$WORK/pip_cache
export TMPDIR=$WORK/tmp
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export XDG_CACHE_HOME=$WORK/.cache
export HF_HOME=$WORK/hf_cache
export PYTHONNOUSERSITE=1  # ignore ~/.local packages
mkdir -p $PIP_CACHE_DIR $TMPDIR $CONDA_PKGS_DIRS $XDG_CACHE_HOME $HF_HOME

# Create directory structure
mkdir -p $EFFOCR_DIR/{gold_data,training_data/char,training_data/word,output,models}

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

    # EffOCR from fork — install WITHOUT deps to avoid pip overwriting conda's torch
    pip install --no-deps git+https://github.com/nealcaren/efficient_ocr.git

    # Install EffOCR's non-torch deps separately
    pip install timm faiss-cpu pytorch-metric-learning onnxruntime onnx \
        opencv-python-headless scipy pandas albumentations kornia \
        huggingface_hub transformers safetensors fonttools wandb

    # Re-pin conda torch (pip deps may have overwritten it)
    conda install --yes -c pytorch -c nvidia pytorch==2.5.1 torchvision pytorch-cuda=12.1 --force-reinstall

    # Remove brotli — causes httpx DecodingError when downloading from HuggingFace
    pip uninstall brotlicffi brotli -y 2>/dev/null || true

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

# --- Download gold-standard data from HuggingFace ---
echo ""
echo "=== Downloading gold-standard data from HuggingFace ==="

GOLD_DIR=$EFFOCR_DIR/gold_data

python -c "
from huggingface_hub import hf_hub_download
import tarfile, os

repo = 'NealCaren/newspaper-ocr-gold'
gold_dir = '$GOLD_DIR'
os.makedirs(gold_dir, exist_ok=True)

# Download verified labels
print('Downloading verified_lines.jsonl...')
hf_hub_download(repo, 'verified_lines.jsonl', repo_type='dataset', local_dir=gold_dir)

# Download sample metadata
print('Downloading sample_metadata.json...')
hf_hub_download(repo, 'sample_metadata.json', repo_type='dataset', local_dir=gold_dir)

# Download and extract image archives per split
for split in ['train', 'val', 'test']:
    fname = f'{split}_images.tar.gz'
    split_dir = os.path.join(gold_dir, split)
    if os.path.isdir(split_dir):
        print(f'{split}/ already extracted, skipping')
        continue
    print(f'Downloading {fname}...')
    path = hf_hub_download(repo, fname, repo_type='dataset')
    print(f'Extracting {fname}...')
    with tarfile.open(path) as tar:
        tar.extractall(gold_dir)
    print(f'  Extracted to {split_dir}')

print('Gold data download complete.')
"

# Count what we got
echo ""
echo "Gold data summary:"
for split in train val test; do
    if [ -d "$GOLD_DIR/$split" ]; then
        pages=$(ls -d $GOLD_DIR/$split/*/ 2>/dev/null | wc -l)
        images=$(find $GOLD_DIR/$split -name "*.png" | wc -l)
        echo "  $split: $pages pages, $images line images"
    fi
done
lines=$(wc -l < $GOLD_DIR/verified_lines.jsonl 2>/dev/null || echo 0)
echo "  Total verified lines: $lines"

# --- Build EffOCR char/word training data from gold labels ---
echo ""
echo "=== Building EffOCR training data from gold labels ==="

python $WORK/ocr-finetune/scripts/effocr/build_training_data.py \
    --gold-jsonl $GOLD_DIR/verified_lines.jsonl \
    --image-dir $GOLD_DIR \
    --output-dir $EFFOCR_DIR/training_data \
    --model-dir $EFFOCR_DIR/models \
    --scales 1.0,0.5,0.35,0.25

echo ""
echo "Setup complete."
echo "  Environment: $WORK/envs/effocr"
echo "  Gold data: $GOLD_DIR/"
echo "  Training data dir: $EFFOCR_DIR/training_data/"
echo "  Output dir: $EFFOCR_DIR/output/"
echo ""
echo "Next step: submit training job:"
echo "  sbatch run_effocr_finetune.sl"

#!/bin/bash
# Set up EffOCR fine-tuning environment on Longleaf.
# Downloads gold-standard data from HuggingFace.
# Training data is built on the compute node (SLURM job), not here.
#
# Usage:
#   cd /work/users/n/c/ncaren/ocr-finetune
#   bash scripts/effocr/setup_longleaf.sh
#
# Then submit:
#   sbatch run_effocr_finetune.sl

set -e

WORK=/work/users/n/c/ncaren
EFFOCR_DIR=$WORK/effocr-finetune

# --- Redirect ALL caches to /work (home has 50GB quota) ---
export PIP_CACHE_DIR=$WORK/pip_cache
export TMPDIR=$WORK/tmp
export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export XDG_CACHE_HOME=$WORK/.cache
export HF_HOME=$WORK/hf_cache
export PYTHONNOUSERSITE=1
mkdir -p $PIP_CACHE_DIR $TMPDIR $CONDA_PKGS_DIRS $XDG_CACHE_HOME $HF_HOME

# --- Create directory structure ---
mkdir -p $EFFOCR_DIR/{gold_data,training_data/char,training_data/word,output,models}

# --- Set up conda ---
module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"

# --- Create or activate conda env ---
if [ -d "$WORK/envs/effocr" ]; then
    echo "Conda env already exists: $WORK/envs/effocr"
    conda activate $WORK/envs/effocr
else
    echo "Creating conda env: $WORK/envs/effocr"
    conda create --yes --prefix $WORK/envs/effocr python=3.11
    conda activate $WORK/envs/effocr

    # Step 1: Install PyTorch with CUDA from conda (the ONLY way to get CUDA libs right)
    echo ""
    echo "=== Installing PyTorch with CUDA 12.1 ==="
    conda install --yes -c pytorch -c nvidia pytorch==2.5.1 torchvision pytorch-cuda=12.1

    # Step 2: Install EffOCR fork WITHOUT deps (--no-deps prevents pip from pulling its own torch)
    echo ""
    echo "=== Installing EfficientOCR fork ==="
    pip install --no-deps git+https://github.com/nealcaren/efficient_ocr.git

    # Step 3: Install EffOCR's non-torch dependencies via pip
    echo ""
    echo "=== Installing EffOCR dependencies ==="
    pip install --no-deps timm  # --no-deps to avoid pulling torch again
    pip install faiss-cpu pytorch-metric-learning onnxruntime onnx \
        opencv-python-headless scipy pandas albumentations kornia \
        huggingface_hub transformers safetensors fonttools wandb

    # Step 4: Remove brotli (causes httpx DecodingError with HuggingFace downloads)
    pip uninstall brotlicffi brotli -y 2>/dev/null || true

    # Step 5: Verify everything works
    echo ""
    echo "=== Verifying installation ==="
    python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
assert not 'libcudnn' in str(torch.__file__) or torch.cuda.is_available(), \
    'CUDA should be available but is not — torch install is broken'
from efficient_ocr import EffOCR
print('EffOCR import OK')
import timm
print(f'timm {timm.__version__}')
print()
print('All imports OK!')
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

# NOTE: Training data (char/word crops) is built on the compute node by the SLURM job.
# This avoids running slow EffOCR inference on the login node.

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "  Environment: $WORK/envs/effocr"
echo "  Gold data:   $GOLD_DIR/"
echo "  Output dir:  $EFFOCR_DIR/output/"
echo ""
echo "Next step — submit the training job:"
echo "  sbatch run_effocr_finetune.sl"
echo ""
echo "The SLURM job will:"
echo "  1. Build training data from gold labels (on GPU node)"
echo "  2. Train char recognizer (50 epochs)"
echo "  3. Train word recognizer (50 epochs)"

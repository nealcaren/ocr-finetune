#!/bin/bash
# Set up EffOCR fine-tuning environment on Longleaf using uv.
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
VENV=$WORK/envs/effocr-uv

# --- Redirect ALL caches to /work (home has 50GB quota) ---
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export HF_HOME=$WORK/hf_cache
export UV_CACHE_DIR=$WORK/uv_cache
export PYTHONNOUSERSITE=1
mkdir -p $TMPDIR $XDG_CACHE_HOME $HF_HOME $UV_CACHE_DIR

# --- Create directory structure ---
mkdir -p $EFFOCR_DIR/{gold_data,training_data/char,training_data/word,output,models}

# --- Install uv if needed ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# --- Create or reuse venv ---
if [ -d "$VENV" ]; then
    echo "Venv already exists: $VENV"
else
    echo "Creating venv: $VENV"
    uv venv --python 3.11 "$VENV"
fi

# Activate
source "$VENV/bin/activate"

# --- Install everything with uv pip ---
echo ""
echo "=== Installing PyTorch with CUDA 12.1 ==="
uv pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Installing EffOCR + dependencies ==="
uv pip install --no-deps "efficient_ocr @ git+https://github.com/nealcaren/efficient_ocr.git"
uv pip install --no-deps timm pytorch-metric-learning
uv pip install faiss-cpu onnxruntime onnx \
    opencv-python-headless scipy pandas albumentations kornia \
    huggingface_hub transformers safetensors fonttools wandb httpx pillow

# --- Verify ---
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
from efficient_ocr import EffOCR
print('EffOCR import OK')
import timm
print(f'timm {timm.__version__}')
print('All imports OK!')
"

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

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "  Environment: $VENV"
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

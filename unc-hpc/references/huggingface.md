# Hugging Face Models on UNC HPC

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Model Caching](#model-caching)
3. [Downloading Models](#downloading-models)
4. [GPU Sizing for Common Models](#gpu-sizing-for-common-models)
5. [Inference Examples](#inference-examples)
6. [SLURM Scripts](#slurm-scripts)
7. [Fine-Tuning](#fine-tuning)
8. [Fine-Tuning with Unsloth](#fine-tuning-with-unsloth)
9. [Troubleshooting](#troubleshooting)

---

## Environment Setup

```bash
module purge
module load anaconda/2024.02

# Create or activate your environment
conda activate ml

# Install Hugging Face stack
pip install transformers accelerate datasets tokenizers sentencepiece
pip install torch torchvision torchaudio  # if not already installed via conda

# For quantized models (lower memory)
pip install bitsandbytes

# For faster inference
pip install optimum

# CLI tool for downloading models
pip install huggingface_hub[cli]
```

## Model Caching

**Critical:** By default, Hugging Face downloads models to `~/.cache/huggingface/` in your home directory. Home quota is only 50GB — a single 7B model in FP16 is ~14GB. Always redirect the cache.

Set these environment variables in every SLURM script and in your `~/.bashrc`:

```bash
# Add to ~/.bashrc AND to every SLURM script
export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_DATASETS_CACHE=/work/users/<o>/<n>/<onyen>/hf_cache/datasets
export TRANSFORMERS_CACHE=/work/users/<o>/<n>/<onyen>/hf_cache/hub
```

Create the directory:
```bash
mkdir -p /work/users/<o>/<n>/<onyen>/hf_cache
```

## Downloading Models

Compute nodes may have limited or no internet access. Download models before submitting GPU jobs.

### Option A: Download on Login Node (small models)

```bash
# Set cache location first
export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache

# Download via CLI
huggingface-cli download bert-base-uncased
huggingface-cli download meta-llama/Llama-2-7b-hf --token YOUR_TOKEN

# Or in Python
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
"
```

### Option B: Download Job (large models)

For large models, submit a CPU job so you don't tie up the login node:

```bash
#!/bin/bash
#SBATCH -J download_model
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH -t 4:00:00
#SBATCH -o download_%j.out

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache

# Download model and tokenizer
huggingface-cli download meta-llama/Llama-2-7b-hf --token YOUR_TOKEN
```

### Option C: Offline Mode

After downloading, force offline mode to avoid network issues on compute nodes:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Gated Models (Llama, Gemma, etc.)

Some models require access approval on huggingface.co. After approval:

```bash
# Login once (stores token in cache)
huggingface-cli login

# Or pass token directly
huggingface-cli download meta-llama/Llama-2-7b-hf --token hf_xxxxx
```

## GPU Sizing for Common Models

Approximate VRAM needed for inference (FP16). Training requires ~2-3x more.

| Model | Parameters | FP16 VRAM | FP32 VRAM | Recommended Partition |
|-------|-----------|-----------|-----------|----------------------|
| BERT-base | 110M | <1 GB | ~1 GB | Any GPU, even `gpu` (GTX 1080) |
| BERT-large | 340M | ~1 GB | ~2 GB | Any GPU |
| RoBERTa-large | 355M | ~1 GB | ~2 GB | Any GPU |
| GPT-2 | 124M | <1 GB | ~1 GB | Any GPU |
| GPT-2 XL | 1.5B | ~3 GB | ~6 GB | `gpu` or above |
| Llama-2-7B | 7B | ~14 GB | ~28 GB | `volta-gpu`, `a100-gpu`, `l40-gpu` |
| Llama-2-13B | 13B | ~26 GB | ~52 GB | `a100-gpu` (40GB) or `l40-gpu` (48GB) |
| Llama-2-70B | 70B | ~140 GB | N/A | `h100_sn` x4 or quantized |
| Mistral-7B | 7B | ~14 GB | ~28 GB | `volta-gpu`, `a100-gpu`, `l40-gpu` |
| Mixtral-8x7B | 47B | ~90 GB | N/A | `h100_sn` x2+ or quantized |
| Phi-2 | 2.7B | ~6 GB | ~11 GB | `gpu` (8GB FP16) or above |
| Sentence-BERT | 110M | <1 GB | ~1 GB | Any GPU |

### Reducing Memory with Quantization

For models that don't fit in GPU memory:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization — ~4x memory reduction
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

| Model | FP16 | 8-bit | 4-bit |
|-------|------|-------|-------|
| 7B | ~14 GB | ~7 GB | ~4 GB |
| 13B | ~26 GB | ~13 GB | ~7 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

## Inference Examples

### Text Classification (BERT)

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="bert-base-uncased", device=0)
results = classifier(["This is great!", "This is terrible."])
print(results)
```

### Text Generation (Llama / Mistral)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain the concept of social stratification:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Batch Inference Over a Dataset

```python
import pandas as pd
from transformers import pipeline

# Load data
df = pd.read_csv("/work/users/x/y/onyen/data/texts.csv")

# Create pipeline with batching
classifier = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    device=0,
    batch_size=32
)

# Run inference
results = classifier(df["text"].tolist())
df["label"] = [r["label"] for r in results]
df["score"] = [r["score"] for r in results]

df.to_csv("/work/users/x/y/onyen/results/classified.csv", index=False)
```

### Embeddings (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
sentences = ["First sentence", "Second sentence", "Third sentence"]
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
# embeddings.shape: (3, 384)
```

Install: `pip install sentence-transformers`

### Zero-Shot Classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
result = classifier(
    "The government announced new climate policies today.",
    candidate_labels=["politics", "environment", "economy", "sports"]
)
print(result["labels"][0], result["scores"][0])
```

## SLURM Scripts

### Basic Inference Job (Small Model)

For BERT-sized models on any GPU:

```bash
#!/bin/bash
#SBATCH -J hf_inference
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -t 4:00:00
#SBATCH -p gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o hf_infer_%j.out

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_HUB_OFFLINE=1

python inference.py
```

### Large Model Inference (7B-13B)

```bash
#!/bin/bash
#SBATCH -J llm_inference
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH -t 1-00:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o llm_%j.out

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_HUB_OFFLINE=1

python inference_llm.py
```

### Very Large Model (70B, Quantized)

```bash
#!/bin/bash
#SBATCH -J llm_70b
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH -t 2-00:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o llm70b_%j.out

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_HUB_OFFLINE=1

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = 'meta-llama/Llama-2-70b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map='auto'
)

# Your inference code here
"
```

### Multi-GPU Inference (Model Parallelism)

For models too large for a single GPU, `device_map='auto'` splits across GPUs:

```bash
#!/bin/bash
#SBATCH -J multi_gpu_infer
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH -t 1-00:00:00
#SBATCH -p h100_sn
#SBATCH --gpus=4
#SBATCH -o mgpu_infer_%j.out

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_HUB_OFFLINE=1

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-70b-hf',
    torch_dtype=torch.float16,
    device_map='auto'  # Automatically splits across all available GPUs
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf')

# Model is now distributed across 4 H100s
print(model.hf_device_map)
"
```

## Fine-Tuning

### LoRA Fine-Tuning (Parameter Efficient)

Install: `pip install peft trl`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="/work/users/x/y/onyen/data/train.jsonl")

training_args = TrainingArguments(
    output_dir="/work/users/x/y/onyen/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
)

trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset["train"], tokenizer=tokenizer)
trainer.train()
model.save_pretrained("/work/users/x/y/onyen/lora_adapter")
```

## Fine-Tuning with Unsloth

[Unsloth](https://github.com/unslothai/unsloth) fine-tunes LLMs 2-5x faster with 70-80% less VRAM than standard HF training, with zero accuracy loss. Highly recommended for fine-tuning on cluster GPUs.

### Install Unsloth

```bash
module purge
module load anaconda/2024.02
conda activate ml

# Basic install
pip install unsloth

# For Ampere+ GPUs (A100, H100, L40/L40S, RTX 3090+) — use the ampere variant:
# Match your torch + CUDA versions. Check with: python -c "import torch; print(torch.__version__)"
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For older GPUs (V100, GTX 1080) — standard variant:
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

GPU compatibility: V100 and above (CUDA Capability >= 7.0). Use `-ampere` install for A100/H100/L40S; standard install for V100/GTX 1080.

### Unsloth VRAM Requirements (QLoRA 4-bit)

Much lower than standard HF training. These are minimums with Unsloth:

| Model Size | Unsloth QLoRA (4-bit) | Unsloth LoRA (16-bit) | Recommended Partition |
|-----------|----------------------|----------------------|----------------------|
| 3B | 3.5 GB | 8 GB | Any GPU |
| 7-8B | 5-6 GB | 19-22 GB | `gpu` (4-bit) or `volta-gpu`+ (16-bit) |
| 13-14B | 8.5 GB | 33 GB | `volta-gpu` (4-bit) or `l40-gpu` (16-bit) |
| 27B | 22 GB | 64 GB | `l40-gpu` (4-bit) or `h100_sn` (16-bit) |
| 70B | 41 GB | 164 GB | `l40-gpu` (4-bit) or `h100_sn` x4 (16-bit) |

Comparison: Standard HF training of Llama 8B needs ~9+ GB VRAM even with Flash Attention 2. Unsloth fits it in ~6 GB with QLoRA.

### Basic Unsloth Fine-Tuning

```python
from unsloth import FastLanguageModel
import torch

# 1. Load model (4-bit quantized)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # or any HF model ID
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,                        # 0 is optimized
    bias="none",                           # "none" is optimized
    use_gradient_checkpointing="unsloth",  # 30% less VRAM (note: string, not bool)
)

# 3. Train with SFTTrainer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="/work/users/x/y/onyen/data/train.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="/work/users/x/y/onyen/outputs",
        report_to="none",
    ),
)

trainer.train()
```

### Chat/Conversational Fine-Tuning

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Format dataset as conversations
def format_chat(examples):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["conversations"]
    ]
    return {"text": texts}

dataset = dataset.map(format_chat, batched=True)
# Then add LoRA and train with SFTTrainer as above
```

### Saving Models

```python
# Save LoRA adapter only (small, fast)
model.save_pretrained("/work/users/x/y/onyen/lora_adapter")
tokenizer.save_pretrained("/work/users/x/y/onyen/lora_adapter")

# Save merged model (LoRA merged into base weights, 16-bit)
model.save_pretrained_merged("/work/users/x/y/onyen/merged_model", tokenizer, save_method="merged_16bit")

# Export to GGUF (for llama.cpp / Ollama)
model.save_pretrained_gguf("/work/users/x/y/onyen/gguf_model", tokenizer, quantization_method="q4_k_m")

# Push to Hugging Face Hub
model.push_to_hub("your-username/model-name", token="hf_...")
tokenizer.push_to_hub("your-username/model-name", token="hf_...")
```

If OOM during merge/save: `model.save_pretrained_merged(..., maximum_memory_usage=0.5)`

### Inference After Fine-Tuning

```python
FastLanguageModel.for_inference(model)  # Enables 2x faster inference

messages = [{"role": "user", "content": "What is machine learning?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### SLURM Script: Unsloth Fine-Tuning on Longleaf

```bash
#!/bin/bash
#SBATCH -J unsloth_ft
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48g
#SBATCH -t 1-00:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o unsloth_%j.out
#SBATCH -e unsloth_%j.err

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi

python finetune.py

echo "Done at $(date)"
```

### SLURM Script: Unsloth on Sycamore H100

```bash
#!/bin/bash
#SBATCH -J unsloth_h100
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 1-00:00:00
#SBATCH -p h100_sn
#SBATCH --gpus=1
#SBATCH -o unsloth_h100_%j.out
#SBATCH -e unsloth_h100_%j.err

module purge
module load anaconda/2024.02
conda activate ml

export HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache
export HF_HUB_OFFLINE=1

python finetune.py
```

### Unsloth Gotchas

- **`use_gradient_checkpointing="unsloth"`** is a string, not a boolean. `True` uses standard PyTorch checkpointing (less efficient). The string `"unsloth"` activates the optimized version.
- **`lora_dropout=0` and `bias="none"`** enable Unsloth's fast code paths. Other values work but fall back to slower implementations.
- **Use `bf16=True` on A100/H100/L40S**, not `fp16`. The auto-detect pattern `bf16=torch.cuda.is_bf16_supported()` handles this. Use `fp16=True` only on V100/GTX 1080.
- **Single-GPU optimized.** Unsloth is primarily designed for single-GPU training. For multi-GPU, standard FSDP/DeepSpeed may be more appropriate.
- **Downloads can stall.** If Unsloth downloads get stuck at 90-95%, set `os.environ['UNSLOTH_STABLE_DOWNLOADS'] = "1"` before importing.
- **Merging can OOM.** Use `maximum_memory_usage=0.5` when calling `save_pretrained_merged` on memory-constrained GPUs.
- **Unsloth provides pre-quantized models** on HF at `huggingface.co/unsloth` (e.g., `unsloth/Llama-3.2-3B-Instruct`). These load faster and use the optimal quantization for Unsloth.

---

## Troubleshooting

**"CUDA out of memory"**
- Reduce batch size
- Use FP16: `torch_dtype=torch.float16`
- Use quantization: `BitsAndBytesConfig(load_in_4bit=True)`
- Use a larger GPU partition
- For inference: `torch.no_grad()` context manager

**"OSError: Can't load tokenizer / model"**
- Model not downloaded yet. Download first (see above).
- Set `HF_HOME` correctly.
- For gated models: run `huggingface-cli login` first.

**"No space left on device"**
- Cache is filling home directory. Set `HF_HOME` to `/work`.
- Check quota: `quota` or visit `https://service.rc.unc.edu/`

**Very slow model loading**
- Loading from home directory instead of `/work`. Set `HF_HOME` to `/work`.
- Network download happening on compute node. Pre-download and use `HF_HUB_OFFLINE=1`.

**"RuntimeError: Expected all tensors to be on the same device"**
- Use `device_map="auto"` when loading model, or explicitly `.to("cuda")` both model and inputs.

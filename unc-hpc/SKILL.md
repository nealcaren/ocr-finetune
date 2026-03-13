---
name: unc-hpc
description: "Run computational jobs on UNC Chapel Hill's HPC clusters (Longleaf and Sycamore). Use when a user wants to: submit SLURM batch jobs, set up Python/conda environments for ML/deep learning, run PyTorch or TensorFlow training on GPUs, run Hugging Face model inference or fine-tuning (including Unsloth for fast QLoRA), use Jupyter notebooks on the cluster, transfer data, manage storage, monitor GPU utilization, write SLURM scripts, do hyperparameter sweeps, or anything involving UNC Research Computing infrastructure. Covers both Longleaf (high-throughput, broad GPU selection) and Sycamore (HPC, H100 GPUs)."
---

# UNC HPC Clusters

Guide for running ML/computational jobs on UNC Chapel Hill's Longleaf and Sycamore clusters.

## Cluster Selection

1. Determine which cluster fits the workload:

   **Longleaf** (`longleaf.unc.edu`) — Choose when:
   - Running single-node GPU jobs (most ML training)
   - Need interactive Jupyter via Open OnDemand (`ondemand.rc.unc.edu`)
   - Want broadest GPU selection (L40S 48GB, A100 40GB, V100 16GB, GTX 1080 8GB)
   - Running many independent jobs (sweeps, preprocessing)

   **Sycamore** (`sycamore.unc.edu`) — Choose when:
   - Need H100 GPUs (80GB HBM3) for large model training
   - Running multi-node distributed training (InfiniBand NDR)
   - Need massive CPU parallelism (192 cores/node, 1.5TB RAM/node)
   - Running tightly coupled MPI workloads

2. See cluster-specific details:
   - **Longleaf partitions, GPUs, storage**: Read [references/longleaf.md](references/longleaf.md)
   - **Sycamore partitions, GPUs, storage**: Read [references/sycamore.md](references/sycamore.md)
   - **Ready-to-use SLURM scripts**: Read [references/slurm-examples.md](references/slurm-examples.md)
   - **Hugging Face models** (inference, fine-tuning, caching, GPU sizing): Read [references/huggingface.md](references/huggingface.md)

## Core Workflow

All jobs on both clusters follow this pattern:

### 1. Connect

```bash
# Off-campus: connect VPN first (Cisco AnyConnect -> vpn.unc.edu, group UNCCampus)
ssh <onyen>@longleaf.unc.edu   # or sycamore.unc.edu
```

Longleaf also has web access: `https://ondemand.rc.unc.edu` (Jupyter, RStudio, shell, file browser). Sycamore is CLI-only.

### 2. Set Up Environment

```bash
module purge
module load anaconda/2024.02

# First time: create conda env
conda create --name ml python=3.12
conda activate ml
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
# Or for TensorFlow: conda install -c conda-forge tensorflow

# Register Jupyter kernel (Longleaf OnDemand only, requires conda not venv)
python -m ipykernel install --user --name=ml
```

### 3. Stage Data

Place training data on `/work` (high-throughput SSD storage, shared across both clusters):

```bash
# Path pattern: /work/users/<first-char>/<second-char>/<onyen>/
# Example for onyen "goheels": /work/users/g/o/goheels/
rsync -avz ./data/ <onyen>@rc-dm.its.unc.edu:/work/users/x/y/<onyen>/data/
```

Use data mover nodes (`rc-dm.its.unc.edu`) for transfers, not login nodes. Use Globus for large transfers (>10 min).

### 4. Write and Submit SLURM Script

Minimal GPU job on Longleaf:

```bash
#!/bin/bash
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 1-00:00:00
#SBATCH -o train_%j.out

module purge
module load anaconda/2024.02
conda activate ml
python train.py
```

Minimal GPU job on Sycamore:

```bash
#!/bin/bash
#SBATCH -p h100_sn
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 1-00:00:00
#SBATCH -o train_%j.out

module purge
module load anaconda/2024.02
conda activate ml
python train.py
```

Key differences: Longleaf uses `--gres=gpu:N` + `--qos=gpu_access`; Sycamore uses `--gpus=N` (no qos needed for `h100_sn`).

Submit: `sbatch train.sl`

### 5. Monitor

```bash
squeue -u <onyen>           # Job status
seff <jobid>                # Efficiency report (after completion)
ssh <nodename>              # SSH to compute node, then:
nvidia-smi                  # GPU snapshot
nvtop                       # Real-time GPU monitor
```

## Key Differences at a Glance

| | Longleaf | Sycamore |
|---|---|---|
| SSH host | `longleaf.unc.edu` | `sycamore.unc.edu` |
| Web portal | OnDemand (`ondemand.rc.unc.edu`) | None |
| Best GPUs | L40S (48GB), A100 (40GB) | H100 (80GB) |
| GPU request | `--gres=gpu:N` + `--qos=gpu_access` | `--gpus=N` |
| Min cores (default) | 1 | 48 (use `-p small` for <48) |
| Multi-node GPU | No | Yes (InfiniBand, `h100_mn`) |
| Max walltime | 11 days (6 for A100) | 5 days |

## Storage (Shared Across Both Clusters)

| Path | Quota | Backed Up | Notes |
|------|-------|-----------|-------|
| `/nas/longleaf/home/<onyen>` | 50GB | Yes | Scripts, configs. Avoid heavy I/O. |
| `/users/<o>/<n>/<onyen>` | 10TB | No | Inactive datasets, capacity expansion |
| `/work/users/<o>/<n>/<onyen>` | 10TB | No | **Active computation data. Use this for training.** |
| `/pine/scr/<o>/<n>/<onyen>` | 30TB | No | Scratch. **36-day purge policy.** |
| `/proj/<labname>` | 1TB+ | No | PI shared space (request via research@unc.edu) |

`<o>` and `<n>` are first and second chars of your ONYEN.

## Quick GPU Selection Guide

| Task | Recommended GPU | Cluster | Partition |
|------|----------------|---------|-----------|
| Quick test / small model | GTX 1080 (8GB) | Longleaf | `gpu` |
| Standard training (FP32) | L40S (48GB) | Longleaf | `l40-gpu` |
| Large model / need FP64 | A100 (40GB) | Longleaf | `a100-gpu` |
| Largest models / fastest | H100 (80GB) | Sycamore | `h100_sn` |
| Multi-node distributed | H100 x8 (640GB) | Sycamore | `h100_mn` |
| Interactive Jupyter w/ GPU | A100 MIG slice | Longleaf | OnDemand |

## Common Pitfalls

- **Never run jobs on login nodes** — always use `sbatch` or `srun`
- **Don't train from home directory** — use `/work` for data I/O
- **Don't transfer via login nodes** — use `rc-dm.its.unc.edu`
- **Start with 1 GPU** — multi-GPU often underutilizes without proper DDP setup
- **Don't mix pip and conda extensively** — install conda packages first, pip last
- **Always `module purge` at top of SLURM scripts** — prevents environment conflicts
- **venv won't work with OnDemand Jupyter** — use conda for Jupyter kernels
- **Scratch purges after 36 days** — don't store important results on `/pine/scr`
- **Sycamore `batch` partition minimum is 48 cores** — use `-p small` for smaller jobs
- **HF models fill home directory** — set `HF_HOME=/work/users/<o>/<n>/<onyen>/hf_cache` to avoid 50GB quota

## Getting Help

- Email: `research@unc.edu`
- Docs: `https://help.rc.unc.edu/`
- Account requests: `https://tdx.unc.edu/TDClient/33/Portal/Requests/ServiceDet?ID=45`
- Quota check: `https://service.rc.unc.edu/` or run `quota`

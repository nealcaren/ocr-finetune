# Longleaf Cluster Reference

## Overview

Longleaf is UNC's **high-throughput computing (HTC)** cluster: ~6,500 cores, 300+ nodes, ~280 GPUs. Free for all UNC researchers. Optimized for many single-node jobs, data science, and GPU computing.

- SSH: `ssh <onyen>@longleaf.unc.edu`
- Web: `https://ondemand.rc.unc.edu` (Jupyter, RStudio, shell, file browser)
- Scheduler: SLURM with fair-share

## GPU Partitions

| Partition | GPU | Count | VRAM | Precision | Max Time | NVLink |
|-----------|-----|-------|------|-----------|----------|--------|
| `gpu` | GTX 1080 | 32 | 8 GB | Single (FP32) | 11 days | No |
| `volta-gpu` | V100 | 80 | 16 GB | Double (FP64) | 11 days | Yes |
| `a100-gpu` | A100 | 24 | 40 GB | Double (FP64) | 6 days | No |
| `l40-gpu` | L40/L40S | 132 | 48 GB | Single (FP32) | 11 days | No |

All GPU jobs require:
```
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:<count>
```

Multi-partition submission (when GPU type doesn't matter):
```
#SBATCH -p a100-gpu,l40-gpu
```

The `l40-gpu` partition has the most GPUs (132), so jobs typically start fastest there.

## CPU Partitions

| Partition | Purpose | Max Time | Notes |
|-----------|---------|----------|-------|
| `general` | Standard jobs (default) | 11 days | Omit `-p` to let SLURM auto-select |
| `bigmem` | Very large memory | 11 days | Requires approval (email research@unc.edu) |
| `interact` | Interactive sessions | 8 hours | For debugging, exploration |
| `datamover` | File transfers | — | Not for computation |

## GPU Selection Tips

- **Most ML training**: `l40-gpu` — 48GB VRAM, plenty for most models, most available
- **Need double precision (FP64)**: `a100-gpu` or `volta-gpu`
- **Quick tests / small models**: `gpu` (GTX 1080, 8GB)
- **Large batch sizes / large models**: `a100-gpu` (40GB) or `l40-gpu` (48GB)

## MIG Slices (OnDemand Jupyter Only)

A100s can be partitioned into smaller slices for interactive work:

| Slice | VRAM | Request |
|-------|------|---------|
| MIG 5GB | 5 GB | `--gres=gpu:1g.5gb:1` |
| MIG 10GB | 10 GB | `--gres=gpu:2g.10gb:1` |
| Full A100 | 40 GB | 2-hour limit per user |

## Open OnDemand

URL: `https://ondemand.rc.unc.edu`

Provides:
- **Jupyter Notebook** with GPU options (MIG slices or full A100)
- **RStudio Server**
- **Remote Desktop** (VNC)
- **Shell access** (web terminal)
- **File browser** (upload/download < 10GB)

To make a conda env available in OnDemand Jupyter:
```bash
conda activate ml
python -m ipykernel install --user --name=ml --display-name "ML Environment"
```

Note: `venv` environments do NOT work with OnDemand Jupyter — must use conda.

## Interactive Sessions

```bash
# CPU interactive
srun -t 5:00:00 -p interact -n 1 --cpus-per-task=1 --mem=8g --pty /bin/bash

# GPU interactive
srun -t 4:00:00 -p a100-gpu --qos=gpu_access --gres=gpu:1 \
     -n 1 --cpus-per-task=8 --mem=32g --pty /bin/bash
```

## Module System

```bash
module avail              # List all modules
module spider <name>      # Search including hidden modules
module load anaconda/2024.02
module load cuda/12.9
module load python/3.12.4
module purge              # Clean slate (always do this in SLURM scripts)
```

Key ML-relevant modules: `anaconda/2024.02`, `python/3.12.x`, `cuda/*`, `cudnn/*`, `r/4.4.0`

## SLURM Quick Reference

```bash
sbatch job.sl             # Submit batch job
srun ...                  # Interactive/inline job
squeue -u <onyen>         # Check your jobs
scancel <jobid>           # Cancel job
seff <jobid>              # Efficiency report (completed jobs)
sacct -j <jobid>          # Accounting info
sinfo                     # Partition/node status
```

### Common SBATCH Directives

```bash
#SBATCH -J jobname              # Job name
#SBATCH -n 1                    # Number of tasks
#SBATCH --cpus-per-task=8       # CPUs per task
#SBATCH --mem=32g               # Total memory
#SBATCH -t 2-00:00:00           # Time (D-HH:MM:SS)
#SBATCH -p l40-gpu              # Partition
#SBATCH --qos=gpu_access        # Required for GPU partitions
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH -o out_%j.out           # Stdout (%j = job ID)
#SBATCH -e err_%j.err           # Stderr
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=<onyen>@email.unc.edu
```

### Array Jobs

```bash
#SBATCH --array=0-9             # 10 tasks (indices 0-9)
```
Access index in script: `$SLURM_ARRAY_TASK_ID`. Output files: `slurm-%A_%a.out`.

### Inline Submission

```bash
sbatch -n 1 --mem=5g -t 1- --wrap="python myscript.py"
```

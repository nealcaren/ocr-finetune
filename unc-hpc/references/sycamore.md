# Sycamore Cluster Reference

## Overview

Sycamore is UNC's **high-performance computing (HPC)** cluster, deployed January 2025. Optimized for large, multi-node, tightly coupled parallel jobs. Features H100 GPUs and 400 Gbps InfiniBand NDR.

- SSH: `ssh <onyen>@sycamore.unc.edu`
- Web portal: None (CLI only)
- Scheduler: SLURM
- Account request: Same portal as Longleaf, select "Sycamore Cluster"

## Hardware

**CPU Nodes (78 total):**

| Type | Count | Cores/Node | RAM/Node | Notes |
|------|-------|------------|----------|-------|
| AMD EPYC 9654 (Genoa) | 40 | 192 (2x96) | 1.5 TB | Standard |
| AMD EPYC 9684X (Genoa-X) | 38 | 192 (2x96) | 1.5 TB | 3D V-Cache, better for some workloads |

Total: ~15,000 cores, 117 TB RAM. All water-cooled.

**GPU Nodes (7 total):**

| Spec | Details |
|------|---------|
| GPU | NVIDIA H100 80GB HBM3 |
| GPUs per node | 4 |
| Total H100s | 28 |
| Intra-node | NVLink (4 GPUs) |
| CPU per GPU node | 2x AMD Bergamo 128-core (256 cores total) |
| RAM per GPU node | 1.5 TB |
| Network | NDR InfiniBand, dual 800 Gbps ports |
| Multi-node | 2 nodes fully IB-interconnected (8 GPUs, 640GB GPU memory) |

## Partitions

| Partition | Min Cores | Max Cores/User | Max Time | Notes |
|-----------|-----------|----------------|----------|-------|
| `batch` (default) | 48 | 5,000 | 5 days | Multi-node MPI jobs |
| `small` | 2 | 192 | 5 days | Single-node jobs <48 cores |
| `lowpri` | 48 | 5,000 | 5 days | May be suspended for higher-priority work |
| `inter` | 1 | 96 | 8 hours | Interactive sessions |
| `h100_sn` | — | 20 GPUs | 5 days | Single-node H100 GPU (up to 4 per node) |
| `h100_mn` | — | 8 GPUs | 5 days | Multi-node H100 (requires special QoS) |

### GPU Job Syntax

Single-node H100 (available to all Sycamore users):
```bash
#SBATCH -p h100_sn
#SBATCH --gpus=1        # Up to 4 per node
```

Multi-node H100 (requires special QoS — request via research@unc.edu):
```bash
#SBATCH -p h100_mn
#SBATCH --gpus=8        # 2 nodes x 4 GPUs
```

Note: Sycamore uses `--gpus=N`, not `--gres=gpu:N` like Longleaf.

### CPU Selection Constraints

Select specific CPU type when it matters:
```bash
#SBATCH -C 9684x          # Hard requirement: V-Cache nodes only
#SBATCH --prefer 9684x    # Soft preference: V-Cache preferred
#SBATCH -C ndr             # Require InfiniBand NDR connectivity
```

## Key Differences from Longleaf

| Feature | Longleaf | Sycamore |
|---------|----------|----------|
| GPU request syntax | `--gres=gpu:N` + `--qos=gpu_access` | `--gpus=N` |
| Default partition min cores | 1 | 48 |
| Small jobs partition | `general` (default) | `-p small` (2-192 cores) |
| Interactive partition | `interact` (8hr) | `inter` (8hr) |
| OnDemand web portal | Yes | No |
| Multi-node GPU | No | Yes (`h100_mn`) |
| InfiniBand | No | Yes (400 Gbps NDR) |

## Storage

Sycamore shares the same storage as Longleaf:
- Home: `/nas/longleaf/home/<onyen>` (50GB, backed up)
- Users: `/users/<o>/<n>/<onyen>` (10TB)
- Work: `/work/users/<o>/<n>/<onyen>` (10TB, high-performance VAST with IB)
- Proj: `/proj/<labname>` (PI-managed)

Tip: Create `~/longleaf/` and `~/sycamore/` subdirectories to keep cluster-specific files organized.

`/work` on Sycamore is backed by the VAST storage system with 8 NDR InfiniBand connections — the fastest storage option for active data.

## Interactive Sessions

```bash
# CPU interactive
srun -t 5:00:00 -p inter -n 1 --cpus-per-task=1 --mem=8g --pty /bin/bash

# GPU interactive
srun -t 4:00:00 -p h100_sn --gpus=1 -n 1 --cpus-per-task=8 --mem=32g --pty /bin/bash
```

## Module System

Same Lmod system as Longleaf. Known available modules include:
- `python/3.13`
- `anaconda/2024.02`
- `r/4.4.2`
- `matlab/2024b`
- `openmpi_5.0.5/gcc_11.4.1`

MPI module naming: `<MPI>_<version>/<Compiler>_<version>`

## When to Use Sycamore vs Longleaf for ML

**Use Sycamore when:**
- Training very large models (LLMs, large vision models) that benefit from H100s
- Need >40GB GPU memory (H100 has 80GB)
- Running distributed training across multiple nodes
- Need fastest possible single-GPU performance

**Use Longleaf when:**
- Running standard single-GPU training
- Want Jupyter via OnDemand
- Running many independent jobs (sweeps, preprocessing)
- Need interactive GPU sessions with web interface
- Model fits in 48GB or less (L40S)

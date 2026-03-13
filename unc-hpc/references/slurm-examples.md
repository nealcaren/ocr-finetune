# SLURM Script Examples

Ready-to-use templates for common ML tasks. Replace `<onyen>`, path characters, and script names.

## Table of Contents

1. [Single-GPU PyTorch (Longleaf)](#single-gpu-pytorch-longleaf)
2. [Single-GPU PyTorch (Sycamore H100)](#single-gpu-pytorch-sycamore-h100)
3. [Single-GPU TensorFlow (Longleaf)](#single-gpu-tensorflow-longleaf)
4. [Multi-GPU DDP (Longleaf)](#multi-gpu-ddp-longleaf)
5. [Multi-Node H100 (Sycamore)](#multi-node-h100-sycamore)
6. [Hyperparameter Sweep Array Job](#hyperparameter-sweep-array-job)
7. [CPU Preprocessing](#cpu-preprocessing)
8. [Flexible Multi-Partition](#flexible-multi-partition)
9. [Conda Environment Setup Job](#conda-environment-setup-job)
10. [Jupyter via SSH Tunnel](#jupyter-via-ssh-tunnel)

---

## Single-GPU PyTorch (Longleaf)

```bash
#!/bin/bash
#SBATCH -J pytorch_train
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32g
#SBATCH -t 2-00:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=<onyen>@email.unc.edu

module purge
module load anaconda/2024.02

conda activate ml

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi

python train.py \
    --epochs 100 \
    --batch-size 64 \
    --data-dir /work/users/x/y/<onyen>/data \
    --output-dir /work/users/x/y/<onyen>/results

echo "Job finished at $(date)"
```

## Single-GPU PyTorch (Sycamore H100)

```bash
#!/bin/bash
#SBATCH -J h100_train
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 2-00:00:00
#SBATCH -p h100_sn
#SBATCH --gpus=1
#SBATCH -o h100_%j.out
#SBATCH -e h100_%j.err

module purge
module load anaconda/2024.02

conda activate ml

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi

python train.py \
    --epochs 100 \
    --batch-size 128 \
    --data-dir /work/users/x/y/<onyen>/data \
    --output-dir /work/users/x/y/<onyen>/results

echo "Job finished at $(date)"
```

## Single-GPU TensorFlow (Longleaf)

```bash
#!/bin/bash
#SBATCH -J tf_train
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40g
#SBATCH -t 3-00:00:00
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o tf_%j.out
#SBATCH -e tf_%j.err

module purge
module load anaconda/2024.02

conda activate tf_env

export TF_FORCE_GPU_ALLOW_GROWTH=true

python train_tf.py
```

## Multi-GPU DDP (Longleaf)

For PyTorch DistributedDataParallel on a single node with multiple GPUs:

```bash
#!/bin/bash
#SBATCH -J multi_gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64g
#SBATCH -t 4-00:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH -o multigpu_%j.out

module purge
module load anaconda/2024.02

conda activate ml

torchrun --nproc_per_node=2 train_ddp.py
```

## Multi-Node H100 (Sycamore)

Distributed training across 2 nodes (8 H100 GPUs total). Requires special QoS access.

```bash
#!/bin/bash
#SBATCH -J multi_node_h100
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256g
#SBATCH -t 3-00:00:00
#SBATCH -p h100_mn
#SBATCH -o mn_h100_%j.out

module purge
module load anaconda/2024.02

conda activate ml

# Get master node address for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_ddp.py
```

## Hyperparameter Sweep Array Job

Runs 10 independent trials with different hyperparameters:

```bash
#!/bin/bash
#SBATCH -J hp_sweep
#SBATCH --array=0-9
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH -t 1-00:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o sweep_%A_%a.out

module purge
module load anaconda/2024.02

conda activate ml

python hp_search.py --trial-id $SLURM_ARRAY_TASK_ID
```

Output files: `sweep_<arrayJobID>_<taskIndex>.out`

## CPU Preprocessing

```bash
#!/bin/bash
#SBATCH -J preprocess
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH -o preprocess_%j.out

module purge
module load anaconda/2024.02

conda activate ml

python preprocess_data.py \
    --input /pine/scr/x/y/<onyen>/raw_data \
    --output /work/users/x/y/<onyen>/processed \
    --workers 16
```

## Flexible Multi-Partition

Submit to multiple GPU partitions — SLURM picks whichever has availability:

```bash
#!/bin/bash
#SBATCH -J ml_flexible
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32g
#SBATCH -t 2-00:00:00
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o flex_%j.out

module purge
module load anaconda/2024.02

conda activate ml
python train.py
```

## Conda Environment Setup Job

Heavy conda installs can time out on login nodes. Run as a job instead:

```bash
#!/bin/bash
#SBATCH -J conda_setup
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -t 2:00:00
#SBATCH -o conda_setup_%j.out

module purge
module load anaconda/2024.02

conda create --name ml python=3.12 -y
conda activate ml

conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y
conda install -c conda-forge scikit-learn pandas matplotlib jupyterlab -y

python -m ipykernel install --user --name=ml --display-name "ML"

echo "Environment setup complete"
conda list
```

## Jupyter via SSH Tunnel

For Jupyter on a GPU node without OnDemand:

```bash
#!/bin/bash
#SBATCH -J jupyter
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -t 8:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o jupyter_%j.out

module purge
module load anaconda/2024.02

conda activate ml

# Print connection instructions
NODE=$(hostname)
PORT=8888
echo "============================================"
echo "Run this on your local machine:"
echo "ssh -L ${PORT}:${NODE}:${PORT} <onyen>@longleaf.unc.edu"
echo "Then open: http://localhost:${PORT}"
echo "============================================"

jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0
```

After submitting, check the output file for the SSH tunnel command and token URL.

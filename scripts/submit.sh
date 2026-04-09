#!/usr/bin/env bash
# =============================================================================
# Bhaskera — generic SLURM job template
# Usage: sbatch scripts/submit.sh configs/my_experiment.yaml
# =============================================================================
#SBATCH --job-name=bhaskera
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

# ── activate bhaskera env (handles CUDA module reload automatically) ─────────
source "$(dirname "$0")/../bhaskera-activate.sh"

# ── Ray cluster bootstrap ────────────────────────────────────────────────────
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
RAY_PORT=6379

# Start head node
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head \
    --node-ip-address="$HEAD_ADDR" \
    --port=$RAY_PORT \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
    --block &

sleep 5

# Start worker nodes
WORKER_NODES=$(( SLURM_JOB_NUM_NODES - 1 ))
if (( WORKER_NODES > 0 )); then
    srun --nodes=$WORKER_NODES \
         --ntasks=$WORKER_NODES \
         --exclude="$HEAD_NODE" \
        ray start \
        --address="${HEAD_ADDR}:${RAY_PORT}" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
        --block &
fi

sleep 10

# ── run training ─────────────────────────────────────────────────────────────
mkdir -p logs
RAY_ADDRESS="${HEAD_ADDR}:${RAY_PORT}" bhaskera-train --config "$CONFIG"

# ── shutdown ─────────────────────────────────────────────────────────────────
ray stop

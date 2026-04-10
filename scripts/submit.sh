#!/bin/bash
# =============================================================================
# Bhaskera — Ray-on-SLURM submission script (using ray symmetric-run)
#
# Requires Ray >= 2.49
#
# Usage:
#   sbatch scripts/submit.sh --config configs/config.yaml
#   sbatch scripts/submit.sh --config configs/config.yaml --num-workers 8
# =============================================================================

#SBATCH --job-name=bhaskera
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/bhaskera_%j_%N.out
#SBATCH --error=logs/bhaskera_%j_%N.err

set -euo pipefail
EXTRA_ARGS="$*"

# =============================================================================
# Environment
# =============================================================================
source "${SLURM_SUBMIT_DIR}/bhaskera-activate.sh"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# NCCL auto-tuning — detects network type and sets optimal defaults
# =============================================================================
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

# ── Auto-detect network fabric ──────────────────────────────────────────────
if [ -d /sys/class/infiniband ] && ls /sys/class/infiniband/ &>/dev/null; then
    echo "[NCCL] InfiniBand detected"
    export NCCL_IB_DISABLE=0
    IB_DEVICES=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
    if [ -n "$IB_DEVICES" ]; then
        export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_DEVICES}"
    fi
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    if ip link show | grep -q "ib0"; then
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    fi
elif [ -d /sys/class/infiniband ] && ibstat 2>/dev/null | grep -q "Rate"; then
    echo "[NCCL] RoCE detected"
    export NCCL_IB_DISABLE=0
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
else
    echo "[NCCL] No InfiniBand/RoCE found — using TCP sockets"
    export NCCL_IB_DISABLE=1
    ETH_IF=$(ip -o link show up | awk -F': ' '{print $2}' \
             | grep -v -E '^(lo|docker|veth|br-)' | head -1)
    if [ -n "$ETH_IF" ]; then
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$ETH_IF}"
    fi
    export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
    export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
fi

echo "[NCCL] NCCL_IB_DISABLE=$NCCL_IB_DISABLE  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-auto}"

# =============================================================================
# Resolve head node address
# =============================================================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
NUM_GPUS=2
WORKER_COUNT=$(( SLURM_NNODES * NUM_GPUS ))

# Randomize port to avoid conflicts with other users' Ray clusters
port=$(( 6379 + (SLURM_JOB_ID % 1000) ))

# Resolve to IP address — hostnames can cause GCS timeout issues
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
ip_head="${head_ip}:${port}"
export ip_head

# Give Ray more time to start on shared/busy nodes
export RAY_raylet_start_wait_time_s=120

mkdir -p logs

echo "========================================================"
echo "  Bhaskera Ray-on-SLURM (symmetric-run)"
echo "  Job       : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_NNODES  (head: $head_node / $head_ip)"
echo "  GPUs/node : $NUM_GPUS"
echo "  Total GPUs: $WORKER_COUNT"
echo "  Address   : $ip_head"
echo "========================================================"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export RAY_raylet_start_wait_time_s=300
export RAY_ADDRESS="$ip_head"
# =============================================================================
# Launch via symmetric-run
# =============================================================================
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    ray symmetric-run \
        --address "$ip_head" \
        --min-nodes "$SLURM_JOB_NUM_NODES" \
        --num-cpus="$SLURM_CPUS_PER_TASK" \
        --num-gpus="$NUM_GPUS" \
        -- \
        python -u -m bhaskera.launcher.train \
            --num-workers "$WORKER_COUNT" \
            $EXTRA_ARGS

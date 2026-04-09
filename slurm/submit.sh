#!/bin/bash
# =============================================================================
# Bhaskera — Ray-on-SLURM submission script
#
# This single script handles both head-node bootstrap and worker launch.
# ONE task per node (Ray manages GPU processes internally).
#
# Usage:
#   sbatch slurm/submit.sh --config configs/config.yaml
#   sbatch slurm/submit.sh --config configs/config_llama.yaml --num-workers 8
#
# Tune the SBATCH directives below for your cluster.
# =============================================================================

#SBATCH --job-name=bhaskera
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # ONE task per node — Ray manages GPU processes
#SBATCH --gres=gpu:4               # GPUs per node (Ray uses all of them)
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/bhaskera_%j_%N.out
#SBATCH --error=logs/bhaskera_%j_%N.err
#SBATCH --exclusive

set -euo pipefail
EXTRA_ARGS="$*"

# =============================================================================
# Environment  — EDIT THESE FOR YOUR CLUSTER
# =============================================================================
# Option A: spack
# . /home/apps/SPACK/spack/share/spack/setup-env.sh && spack load cuda@12.1

# Option B: module
# module load cuda/12.1 cudnn/8.9

# Activate your venv
source "${BHASKERA_VENV:-/scratch/your-user/bhaskera/.venv}/bin/activate"
export PYTHONPATH="${BHASKERA_ROOT:-/scratch/your-user/bhaskera/src}:${PYTHONPATH:-}"

# =============================================================================
# NCCL tuning for InfiniBand — comment out IB lines for Ethernet clusters
# =============================================================================
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

# =============================================================================
# Ray cluster bootstrap
# =============================================================================
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
RAY_PORT=$(( 6379 + (SLURM_JOB_ID % 1000) ))
RAY_DASHBOARD_PORT=$(( 8265 + (SLURM_JOB_ID % 1000) ))
WORKER_COUNT=$(( SLURM_NNODES * $(echo "${SLURM_GPUS_PER_NODE:-4}" | cut -d: -f2 2>/dev/null || echo 4) ))

export HEAD_NODE RAY_PORT RAY_DASHBOARD_PORT

echo "========================================================"
echo "  Bhaskera Ray-on-SLURM"
echo "  Job       : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_NNODES  (head: $HEAD_NODE)"
echo "  Workers   : $WORKER_COUNT GPUs total"
echo "  Ray port  : $RAY_PORT  |  Dashboard: $RAY_DASHBOARD_PORT"
echo "========================================================"

mkdir -p logs

# Start Ray head on this node (background)
ray start --head \
    --port="$RAY_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --num-gpus="${SLURM_GPUS_PER_NODE:-4}" \
    --block &
RAY_HEAD_PID=$!

# Wait for head to be ready (TCP poll)
python - <<PYEOF
import socket, time, sys
host, port, timeout = "$HEAD_NODE", $RAY_PORT, 120
t0 = time.time()
while time.time() - t0 < timeout:
    try:
        socket.create_connection((host, port), timeout=3).close()
        print(f"Ray head ready ({time.time()-t0:.1f}s)")
        sys.exit(0)
    except OSError:
        time.sleep(2)
print("ERROR: Ray head did not start in time")
sys.exit(1)
PYEOF

# Start Ray workers on remaining nodes
if [ "$SLURM_NNODES" -gt 1 ]; then
    srun --nodes=$(( SLURM_NNODES - 1 )) \
         --ntasks=$(( SLURM_NNODES - 1 )) \
         --relative=1 \
         ray start \
             --address="${HEAD_NODE}:${RAY_PORT}" \
             --num-gpus="${SLURM_GPUS_PER_NODE:-4}" \
             --block &
fi

# Short wait for workers to register
sleep 10

# =============================================================================
# Launch training driver (runs on head node, submits work to Ray cluster)
# =============================================================================
export RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"

python -m bhaskera.launcher.train \
    --num-workers "$WORKER_COUNT" \
    $EXTRA_ARGS

# Cleanup
wait $RAY_HEAD_PID || true
ray stop || true

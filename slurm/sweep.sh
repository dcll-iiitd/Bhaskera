#!/bin/bash
# =============================================================================
# Bhaskera — SLURM array job for hyperparameter sweeps
#
# Each array task picks one config from CONFIGS[] below.
#
# Usage:
#   sbatch slurm/sweep.sh
#   sbatch --array=0-1 slurm/sweep.sh   # only tasks 0 and 1
# =============================================================================

#SBATCH --job-name=bhaskera_sweep
#SBATCH --array=0-2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/sweep_%A_%a_%N.out
#SBATCH --error=logs/sweep_%A_%a_%N.err
#SBATCH --exclusive

# ---- Configs to sweep -------------------------------------------------------
CONFIGS=(
    "configs/config.yaml"
    "configs/config_llama.yaml"
    "configs/config_custom.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID → config: $CONFIG"

# Delegate to submit.sh — it handles Ray bootstrap
bash slurm/submit.sh --config "$CONFIG"

#!/bin/bash
# Slurm job script to run Param2 inference using Bhaskera
# For clusters with 2 GPUs per node (2x A100). For single-node runs.
# Usage example:
#   sbatch scripts/inference.slurm --config configs/param2.yaml --output-file outputs/param2_raw.txt --prompt "Explain attention."

#SBATCH --job-name=bhaskera-param2-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2                   # request 2 GPUs per node (2x A100)
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

set -euo pipefail

mkdir -p logs outputs

# This script forwards all provided arguments to the Bhaskera inference CLI
# Example: sbatch scripts/inference.slurm --config cfg.yaml --output-file out.txt --prompt "..."

ARGS=("$@")

extract_arg() {
  name="$1"; shift
  for a in "$@"; do
    case "$a" in
      "$name")
        continue
        ;;
      "$name"=*)
        echo "${a#*=}"; return 0 ;;
    esac
  done
  prev=""
  for a in "$@"; do
    if [ "$prev" = "$name" ]; then
      echo "$a"; return 0
    fi
    prev="$a"
  done
  return 1
}

CONFIG_VAL=$(extract_arg --config "${ARGS[@]}" || true)
OUT_VAL=$(extract_arg --output-file "${ARGS[@]}" || true)
PROMPT_VAL=$(extract_arg --prompt "${ARGS[@]}" || true)
PROMPT_FILE_VAL=$(extract_arg --prompt-file "${ARGS[@]}" || true)

if [ -z "${CONFIG_VAL}" ]; then
  echo "ERROR: must pass --config <path> to sbatch (e.g. sbatch scripts/inference.slurm --config configs/param2.yaml)"
  exit 1
fi

if [ -z "${OUT_VAL}" ]; then
  echo "ERROR: must pass --output-file <path> to sbatch"
  exit 1
fi

if [ -z "${PROMPT_VAL}" ] && [ -z "${PROMPT_FILE_VAL}" ]; then
  echo "ERROR: pass either --prompt '...' or --prompt-file prompts.txt to sbatch"
  exit 1
fi

# Use SBATCH allocation for GPUs; allow overriding CPU count via SBATCH
CPUS=${SLURM_CPUS_PER_TASK:-8}

export OMP_NUM_THREADS=${CPUS}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Running: srun --ntasks=1 --cpus-per-task=${CPUS} python -u -m bhaskera.launcher.infer ${ARGS[*]}"
# Rely on SBATCH --gres allocation for GPUs; do not pass --gres to srun to avoid cluster-specific errors
srun --ntasks=1 --cpus-per-task=${CPUS} python -u -m bhaskera.launcher.infer "${ARGS[@]}"

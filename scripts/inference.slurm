#!/bin/bash
# Slurm job script to run Param2 inference using Bhaskera
# Usage examples:
#  sbatch scripts/inference.slurm                     # uses defaults
#  CONFIG=configs/param2.yaml PROMPT_FILE=prompts.txt sbatch scripts/inference.slurm

#SBATCH --job-name=bhaskera-param2-infer
mkdir -p logs outputs

# This script forwards all provided arguments to the Bhaskera inference CLI
# Usage: sbatch scripts/inference.slurm --config cfg.yaml --output-file out.txt --prompt "..."

ARGS=("$@")

extract_arg() {
  name="$1"; shift
  for a in "$@"; do
    case "$a" in
      "$name")
        # next token is the value (if present)
        continue
        ;;
      "$name"=*)
        echo "${a#*=}"; return 0 ;;
    esac
  done
  # handle --name value form
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

# Determine GPUs/CPUs (allow overriding via SBATCH envs or explicit vars)
GPUS=${GPUS:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}

export OMP_NUM_THREADS=${CPUS}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Running: srun --gres=gpu:${GPUS} python -u -m bhaskera.launcher.infer ${ARGS[*]}"
srun --gres=gpu:${GPUS} python -u -m bhaskera.launcher.infer "${ARGS[@]}"
if [ -n "${KV_CACHE:-}" ]; then
  ARGS+=(--kv-cache "${KV_CACHE}")
fi
if [ -n "${TEMPERATURE:-}" ]; then
  ARGS+=(--temperature "${TEMPERATURE}")
fi
if [ -n "${TOP_P:-}" ]; then
  ARGS+=(--top-p "${TOP_P}")
fi

if [ -n "${PROMPT_FILE:-}" ]; then
  ARGS+=(--prompt-file "${PROMPT_FILE}" --output-file "${OUTPUT_FILE}")
else
  ARGS+=(--prompt "${PROMPT}" --output-file "${OUTPUT_FILE}")
fi

# Run (use python -m to ensure package-relative import works without installation)
echo "Running: srun --gres=gpu:${GPUS} python -u -m bhaskera.launcher.infer ${ARGS[*]}"
srun --gres=gpu:${GPUS} python -u -m bhaskera.launcher.infer "${ARGS[@]}"

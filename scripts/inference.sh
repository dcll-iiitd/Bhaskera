#!/usr/bin/env bash
# Local helper to run Param2 inference (also usable inside an interactive Slurm session)
# Usage:
#   ./scripts/inference.sh "Explain attention mechanisms."           # single prompt
#   PROMPT_FILE=prompts.txt ./scripts/inference.sh                    # prompt file
#   CONFIG=configs/param2.yaml ./scripts/inference.sh                # custom config

set -euo pipefail

if [ -f .tmpenv/bin/activate ]; then
  source .tmpenv/bin/activate
fi

mkdir -p outputs logs

# Require explicit config and output paths (no defaults)
if [ -z "${CONFIG:-}" ]; then
  echo "ERROR: CONFIG must be set (e.g. CONFIG=configs/param2.yaml)"
  exit 1
fi

if [ -z "${OUTPUT_FILE:-}" ]; then
  echo "ERROR: OUTPUT_FILE must be set (e.g. OUTPUT_FILE=outputs/param2_raw.txt)"
  exit 1
fi

# Prompt can be passed as first positional arg or via PROMPT_FILE env var
PROMPT="${1:-}"
if [ -z "${PROMPT}" ] && [ -z "${PROMPT_FILE:-}" ]; then
  echo "ERROR: provide a prompt as first arg or set PROMPT_FILE=path"
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Build args only from explicitly provided env/args
ARGS=(--config "${CONFIG}" --max-new-tokens 256)

if [ -n "${DEVICE:-}" ]; then
  ARGS+=(--device "${DEVICE}")
fi
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

echo "Running: python -u -m bhaskera.launcher.infer ${ARGS[*]}"
python -u -m bhaskera.launcher.infer "${ARGS[@]}"

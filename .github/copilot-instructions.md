# Copilot instructions for Bhaskera

## Project reality and current state

- This repository is currently serve-first. The root `README.md` and `ARCHITECTURE.md` still describe an older training-oriented codebase.
- For current behavior, prefer `configs/README.md` and the subsystem READMEs under `src/bhaskera/*/` over the root docs.
- The packaged CLI entry point is only `bhaskera-serve` from `pyproject.toml`; training/inference commands exist as modules and are invoked with `python -m bhaskera.launcher.<name>`.

## Setup and installation

Use the project’s documented install flow instead of ad hoc `pip install -r requirements.txt`:

```bash
bash setup.sh
```

For serving work without the full training stack, install only the runtime extras:

```bash
uv pip install -e ".[serve,vllm,gateway]"
```

Local environment notes:

- Activate the local venv with `source .venv/bin/activate`.
- In SLURM jobs, source `bhaskera-activate.sh` and export `PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"`.
- Running from a checkout without installation requires `PYTHONPATH=src`.

## Build, test, and lint commands

There is no lint/typecheck/CI configuration in this repo, and no ruff/mypy/pre-commit setup is present.

The repository’s single targeted test entry point is:

```bash
PYTHONPATH=src python -m pytest src/bhaskera/inference/tests/test_inference.py -v
```

If `pytest` is installed in the active environment, the equivalent direct command is:

```bash
pytest src/bhaskera/inference/tests/test_inference.py -v
```

Do not invent additional lint or CI commands for this repo; the project does not define them.

## High-level architecture

- `configs/*.yaml` are the primary launch definitions for serving runs, for example:

```bash
bhaskera-serve --config configs/serve_test.yaml --ray-address local
```

- `src/bhaskera/serve/` implements the OpenAI-compatible HTTP API on top of Ray Serve + FastAPI. It supports either a vLLM backend or the HF fallback backend and can include the optional Langfuse gateway for auth and observability.
- `src/bhaskera/launcher/` contains the command entrypoints and glue code (`serve.py`, `train.py`, `tokenize.py`, `infer.py`, `diagnostics.py`, `dashboard.py`). These are the best place to trace the actual flow from CLI flag parsing to execution.
- `src/bhaskera/config.py` defines the YAML/dataclass schema used across the project. Config-driven execution is the central pattern.
- `src/bhaskera/data/` handles dataset registration, chat-format registration, and persisted tokenization caches.
- `src/bhaskera/models/`, `distributed/`, `trainer/`, and `inference/` contain the model-loader, sharding, distributed training, and generation logic.
- `src/bhaskera/utils/` holds metrics and logging integrations.

## Key conventions specific to this repository

- Prefer current code and subsystem docs over stale root-level docs; `README.md` and `ARCHITECTURE.md` are historical references, not the canonical source of truth for the current serving-first repo.
- Use YAML-first configuration changes; model and serving settings are intentionally represented in config files rather than hardcoded values.
- When working in training code, preserve established initialization order: load model -> apply Liger kernels -> apply LoRA -> wrap with FSDP2/DDP. This ordering is enforced in the training path and is not a generic optimization pattern.
- Follow the project’s precision rules:
  - FSDP paths keep autocast off unless the config explicitly dictates otherwise.
  - DDP paths use autocast on.
  - LoRA dtype decisions are strategy-gated and must match the active sharding mode.
- Keep checkpoint behavior aligned with the repo’s sentinel-based semantics: sharded DCP writes land in a `.tmp` directory and rank-0 finalizes with `meta.json` + `.complete` only after a successful write.
- For data and chat-format additions, follow the registry pattern in `src/bhaskera/data/`: use `@register`, `@register_raw`, and `@register_format` instead of ad hoc conditionals.
- For distributed training launches, NCCL environment variables and timeouts must be set in the shell before Python starts (for example in `scripts/submit.sh`), not after process startup.

## Practical workflow

When making changes, first inspect the relevant subsystem README and the specific module that owns the behavior, then update the config or registry entry instead of introducing a one-off special case. The repo is intentionally config-driven and registry-driven rather than hardcoded to a single model or execution mode.

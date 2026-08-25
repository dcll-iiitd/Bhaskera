# Repository Guidelines

## Project Structure & Module Organization

Bhaskera is a Python 3.11 package using the `src/` layout. Core code lives in
`src/bhaskera/`: `launcher/` contains CLI entry points, `trainer/` owns the training
loop, `distributed/` implements FSDP2/DDP and checkpoints, and `data/`, `models/`,
`evaluation/`, `inference/`, `plugins/`, and `utils/` provide their named subsystems.
Scenario-specific YAML files live in `configs/`; SLURM launch support is in
`scripts/submit.sh`. The current tests are under `src/bhaskera/inference/tests/`.
Read the relevant package README and `ARCHITECTURE.md` before changing subsystem
contracts.


## Build, Test, and Development Commands

- `bash setup.sh`: detect CUDA, create the environment, and install Bhaskera.
- `pip install -e ".[dev]"`: install an editable CPU-friendly development setup.
- `pytest src/bhaskera/inference/tests/test_inference.py -v`: run the CPU-only suite.
- `ruff check src/`: lint all package code using the configured 100-character limit.
- `bhaskera-tokenize --config configs/tokenize.yaml --split both`: create a reusable
  tokenized dataset cache.
- `bhaskera-train --config configs/qwen.yaml --num-workers 4`: start a configured Ray
  training job.
- `bhaskera-infer --config configs/inference_param2.yaml --prompt "Hello"`: run inference.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, and `from __future__ import annotations`.
Name modules and functions `snake_case`, classes `PascalCase`, and constants
`UPPER_SNAKE_CASE`. Keep optional heavyweight imports lazy so Ray serialization and
minimal installations continue to work. Comments and docstrings should explain why a
constraint exists, not restate the code. Extend registries through their decorators;
avoid architecture-specific model-name checks outside `introspect.py`.

## Testing Guidelines

Pytest is the test framework; name files `test_*.py`, classes `Test*`, and functions
`test_*`. Add focused CPU tests beside the relevant package where practical. There is
no configured coverage threshold, and most distributed paths require a real Ray/GPU
job, so document the config and hardware used for manual validation. Run pytest and
Ruff before submitting.

## Commit & Pull Request Guidelines

Recent history uses short, lowercase, change-focused subjects such as `galore added`
and `throughput calculation change`. Keep the first line concise and imperative; add a
body when behavior, migration, or distributed tradeoffs need explanation. Pull requests
should summarize the change, list validation commands, link related issues, and call out
configuration or checkpoint compatibility. Include logs or screenshots for dashboard,
metrics, or CLI-visible changes, and update documentation and example YAML when public
behavior changes.

# Branches
- This repo has several branches that include the one for FL.

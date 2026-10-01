# `bhaskera.serve.decision` — logit-readout decisions

Answers yes/no (and, for chat models, choice/score) questions about a text with one forward
pass per question, reading the answer-token logits; nothing is generated. Ported from
[feder-cr/jev](https://github.com/feder-cr/jev) @ 5be02a3 (MIT, see `NOTICE`).

## Layout
| Module | Role |
|---|---|
| `schema`, `prompts`, `decisions`, `calibration` | request contract, prompt compiler (shared state prefix), logits → answers, temperatures |
| `wire`, `translate`, `service` | Jev wire format ⇄ native requests; HTTP-free handlers |
| `runtime/` | llama.cpp b11081 ctypes binding and pinned prebuilt downloader |
| `backend`, `engine` | llama.cpp scoring (seq-copy / state-restore branches), single worker thread |
| `registry`, `loader` | `configs/models/gguf.yaml` → verified weights; driver provisioning; replica loading |

## Runtime notes (A6000, driver 580.173.02, CUDA 13.0)
- llama.cpp build in use: cuda (13.4), release b11081 (`llama-b11081-linux-x64-cuda`). It loaded
  and ran on the GPU first time (`device: gpu`, all layers offloaded), so the cuda12 (12.8)
  fallback was not needed or exercised.
- Weights: upstream feder-cr/jev release `jevos-v2` (`jevos-v2-q8_0.gguf`, `jevos-v2-q4_k_m.gguf`);
  the earlier `jevos` release tag no longer exists upstream.
- jevos branch strategy on the GPU: seq-copy, probe max delta 0.0.
- Measured on the A6000 (q8_0): model load 1.9 s; a one-question `/v1/systemone` request
  took `inference;dur=7.5, total;dur=14.5` ms (Server-Timing).
- Fetch without serving: `bhaskera-decision-fetch --config <yaml>`.

## Configuration
`serve.decision` keys (defaults in `src/bhaskera/config.py`):

| Key | Default | Meaning |
|---|---|---|
| `registry` | `configs/models/gguf.yaml` | GGUF registry (model name to verified weights) |
| `models_dir` | `~/.cache/bhaskera/gguf` | where weights are cached |
| `device` | `cuda` | llama.cpp device: cuda / gpu / cpu / auto |
| `ctx` | `8192` | most tokens per question (state + question) |
| `batch_size` | `4` | question branches per micro-batch (1-16) |
| `prefill_chunk` | `512` | llama.cpp ubatch |
| `branch` | `auto` | auto / seq-copy / state-restore |
| `calibration` | none | calibration JSON from `bhaskera-calibrate` |
| `max_ongoing_requests` | `4` | per-replica concurrency cap |
| `batching.enabled` | `false` | cross-request batching (seq-copy models only) |
| `batching.max_batch_size` | `8` | requests per batch |
| `batching.batch_wait_timeout_s` | `0.005` | wait to fill a batch |
| `llama_cpp.accelerator` | `cuda` | cuda / cuda12 / cpu / auto: pinned b11081 build to fetch |
| `llama_cpp.cache_dir` | `~/.cache/bhaskera/llama.cpp` | runtime download cache |
| `llama_cpp.runtime_dir` | none | set by provisioning, or a local build of b11081 |

## Parity
Gate (amended by the user, 2026-09-30): port fidelity, max |dP(yes)| <= 0.01 against upstream
jev on the same device with the same branch strategy. Result: PASS, bit-identical (max 0.0, 0
flips) for q8_0 and q4_k_m. That PASS is Bhaskera with `branch: state-restore` versus upstream
jev on CUDA (state-restore is upstream's own strategy for llama models). The default seq-copy
path drifts up to 0.053 (q8_0) / 0.110 (q4_k_m) versus upstream CUDA, and batching at
concurrency 16 drifts up to 0.052 versus unbatched; these, along with CPU-vs-GPU drift, are
reported numbers, not gates. Details: `benchmarks/decision/parity/REPORT.md`.

## Benchmarks
Results and the BEST_REPLICAS decision (8): `benchmarks/decision/results/SUMMARY.md`. Rerun on
a GPU host with `scripts/decision/run_bench.sh <upstream|replicas|staterestore|quant|gateway|batching>`
(needs `BASE_CONFIG`, `JEV_DIR`, `GGUF_DIR`; see the script header).

## Next ideas
Parked from the design spec (section 10):
1. Cascade: jevos first, escalate uncertain answers to the 7B model.
2. vLLM 2-way classifier head: jevos as an HF sequence classifier, compared with llama.cpp.
3. Packed multi-question attention: N questions in one forward pass with a mask.
4. PRAXIST-style evidence-graph judge: jevos as the fast yes/no oracle.
5. Distillation: train a jevos-style model with Bhaskera's trainer, then serve it here.

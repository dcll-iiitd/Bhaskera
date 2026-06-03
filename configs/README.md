# `configs/` — Serve Configuration Reference

This directory contains the YAML configurations for `bhaskera-serve`.
Pass any of them to the CLI with `--config`:

```bash
bhaskera-serve --config configs/serve_test.yaml --ray-address local
```

---

## Files at a Glance

| File | Model | VRAM Required | Use Case |
|---|---|---|---|
| `serve_test.yaml` | `DeepSeek-R1-Distill-Qwen-7B` | ~16 GB | Development / smoke testing |
| `openbiollm.yaml` | `OpenBioLLM-Llama3-8B` | ~16 GB | Biomedical / clinical chat |
| `param_vllm.yaml` | `Param2-17B-A2.4B-Thinking` | ~34 GB (1× 80 GB) or split over 2× 40 GB | MoE reasoning model |

---

## `serve_test.yaml` — DeepSeek-R1-Distill-Qwen-7B

```yaml
model:
  name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  dtype: "bfloat16"

serve:
  backend: "vllm"
  num_replicas: 1
  port: 0                             # 0 = pick a free port automatically

  vllm:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.88
    max_model_len: 32768              # 32K context — supports long reasoning traces

  gateway:
    enabled: true
    proxy_port: 0                     # 0 = pick a free port automatically
    cloudflared: true

inference:
  max_new_tokens: 16384               # large budget for chain-of-thought outputs
  temperature: 0.6
  top_p: 0.95
  do_sample: true
```

**When to use:** Default development and testing config. The 7B model fits on a
single 16–24 GB GPU. The 32K context window and 16K token generation budget
accommodate the full reasoning trace that DeepSeek-R1 variants produce before
the final answer.

**Hardware:** 1× GPU with ≥16 GB VRAM.

---

## `openbiollm.yaml` — OpenBioLLM-Llama3-8B

```yaml
model:
  name: "aaditya/OpenBioLLM-Llama3-8B"
  dtype: "bfloat16"

serve:
  backend: "vllm"
  num_replicas: 1
  port: 0

  vllm:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.88
    max_model_len: 8192               # sufficient for clinical notes and reports

  gateway:
    enabled: true
    proxy_port: 0
    cloudflared: true

inference:
  max_new_tokens: 256
  temperature: 0.6
  top_p: 0.9
  do_sample: true
```

**When to use:** Biomedical question-answering, clinical note summarisation,
and medical education. Fine-tuned on PubMed, clinical guidelines, and biomedical
corpora via the OpenBioLLM pipeline.

**Hardware:** 1× GPU with ≥16 GB VRAM. Same footprint as `serve_test.yaml`.

---

## `param_vllm.yaml` — Param2-17B-A2.4B-Thinking (MoE)

```yaml
model:
  name: "bharatgenai/Param2-17B-A2.4B-Thinking"
  dtype: "bfloat16"
  trust_remote_code: true             # required for the custom MoE architecture

serve:
  backend: "vllm"
  num_replicas: 1
  port: 0

  ray_actor_options:
    num_gpus: 1                       # increase to 2 if using 40 GB GPUs

  vllm:
    # ⚠️  HARDWARE CHECK
    # A 17B model in bfloat16 takes ~34 GB of VRAM.
    # - 1× 80 GB GPU (A100-80G, H100): no changes needed.
    # - 2× 40 GB GPUs (A6000, A100-40G): set tensor_parallel_size: 2
    #   AND ray_actor_options.num_gpus: 2
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.92
    max_model_len: 4096               # expanded context for reasoning traces

  gateway:
    enabled: true
    proxy_port: 0
    cloudflared: true

inference:
  max_new_tokens: 4096                # must be large — model reasons before answering
  temperature: 0.6
  top_p: 0.95
  do_sample: true
```

**When to use:** Multilingual reasoning tasks with Hindi/Indic language support.
This is a Mixture-of-Experts model — only 2.4B parameters are active per forward
pass despite the 17B total. The `-Thinking` variant produces an explicit
reasoning trace before the final answer; `max_new_tokens: 4096` ensures the
trace is never cut short.

`trust_remote_code: true` is **required** — the Param2 architecture ships custom
model code and is not part of the standard HuggingFace transformers library.

**Hardware — 1× 80 GB GPU (default config)**

No changes needed. A single A100-80G or H100-80G handles the model at
`gpu_memory_utilization: 0.92`.

**Hardware — 2× 40 GB GPUs**

Edit two fields before launching:

```yaml
serve:
  ray_actor_options:
    num_gpus: 2

  vllm:
    tensor_parallel_size: 2
```

---

## Key Fields Explained

| Field | What it controls |
|---|---|
| `serve.port: 0` | All three configs use `0` — the server picks a free port at startup and prints it in the logs |
| `serve.gateway.proxy_port: 0` | Same auto-selection for the Langfuse Gateway port |
| `serve.gateway.cloudflared: true` | Launches `./cloudflared` in the project root and prints a public `trycloudflare.com` URL |
| `serve.vllm.gpu_memory_utilization` | Fraction of GPU VRAM vLLM may use; leave headroom below 1.0 for CUDA kernels |
| `serve.vllm.max_model_len` | Maximum total tokens (prompt + output) the model will process in one request |
| `inference.max_new_tokens` | Default output token budget when the API caller does not set `max_tokens` |

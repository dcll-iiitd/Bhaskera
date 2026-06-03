# `bhaskera.serve` — OpenAI-Compatible LLM Serving

This module exposes any Bhaskera-supported model as a production-ready HTTP API
that is fully compatible with the OpenAI Chat Completions spec. It is built on
**Ray Serve** (HTTP routing + replica management), **FastAPI** (request validation),
and either **vLLM** (high-throughput GPU path) or the **HuggingFace** backend
(compatibility fallback). An optional **Langfuse Gateway** layer adds API-key
authentication, per-user observability, and a public Cloudflare tunnel.

---

## Installation

Install only the serving extras — no training dependencies needed:

```bash
uv pip install -e ".[serve,vllm,gateway]"
```

| Extra | What it pulls in |
|---|---|
| `serve` | `ray[serve]`, `fastapi`, `uvicorn[standard]`, `pydantic>=2.5` |
| `vllm` | `vllm>=0.4.3` — continuous batching, PagedAttention, tensor parallelism |
| `gateway` | `langfuse>=2.0`, `openai>=1.0` — auth gateway + observability |

> **Note:** `vllm` is GPU-only and requires CUDA. Omit it to use the HF fallback
> on CPU or MPS.

---

## Cloudflared Setup

The Bhaskera gateway can expose your local server to the internet via a free
Cloudflare Quick Tunnel — no Cloudflare account required. The binary must be
present as `./cloudflared` in the directory where you run `bhaskera-serve`.

### Install

```bash
curl -L -o cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ./cloudflared
```

### Verify

```bash
./cloudflared --version
# cloudflared version 2025.x.x (built ...)
```

Once Bhaskera starts, the public HTTPS URL is printed to the log:

```
============================  
🌍 PUBLIC GATEWAY URL (Cloudflare):
   https://random-phrase.trycloudflare.com
============================
```

The tunnel dies when `bhaskera-serve` exits. For persistent tunnels create a
named tunnel via the Cloudflare dashboard — but for development the Quick Tunnel
is zero-config.

---

## Quick Start

```bash
# 1. Start the server (port is auto-assigned when port: 0 in the config)
bhaskera-serve --config configs/serve_test.yaml

# 2. Test the health endpoint (replace PORT with the port printed in the logs)
curl http://localhost:PORT/health

# 3. Send a chat request
curl http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}]
  }'
```

---

## CLI Reference

```
bhaskera-serve --config PATH [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config`, `-c` | path | **required** | Path to YAML config file |
| `--host` | str | from config | Override `serve.host` (e.g. `0.0.0.0`) |
| `--port` | int | from config | Override `serve.port` |
| `--backend` | `vllm` / `hf` | from config | Override the serving backend |
| `--num-replicas` | int | from config | Override replica count |
| `--ray-address` | str | `auto` | Ray cluster address. `auto` = attach to local cluster; `local` = spawn a new single-node cluster; `ray://host:port` = remote cluster |
| `--log-level` | `DEBUG/INFO/WARNING/ERROR` | `INFO` | Python logging verbosity |

### Example — override backend and port at launch time

```bash
bhaskera-serve \
  --config configs/param_vllm.yaml \
  --backend vllm \
  --port 9000 \
  --ray-address local \
  --log-level DEBUG
```

---

## API Reference

### `GET /health`

Liveness probe. Returns 200 as soon as the replica finishes loading the model.

**Response**

```json
{
  "status": "ok",
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "backend": "vllm"
}
```

---

### `GET /v1/models`

List the currently served model, matching the OpenAI `/v1/models` spec.

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      "object": "model",
      "created": 1750000000,
      "owned_by": "bhaskera"
    }
  ]
}
```

---

### `POST /v1/chat/completions`

Main inference endpoint, fully compatible with the OpenAI Chat Completions API.

**Request body**

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | required | Model identifier (any string; for routing label only) |
| `messages` | `array` | required | List of `{role, content}` objects. `role` is `system`, `user`, `assistant`, or `tool` |
| `temperature` | `float` | `1.0` | Sampling temperature `[0.0, 2.0]`. `0.0` = greedy |
| `top_p` | `float` | `1.0` | Nucleus sampling probability `[0.0, 1.0]` |
| `top_k` | `int` | `50` | Top-K sampling (≥1) |
| `max_tokens` | `int` | config default | Max tokens to generate. Falls back to `inference.max_new_tokens` when omitted |
| `stream` | `bool` | `false` | Enable SSE streaming |
| `stop` | `string / string[]` | `null` | Up to 4 stop sequences |
| `presence_penalty` | `float` | `0.0` | Presence penalty `[-2.0, 2.0]` (honoured by vLLM) |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty `[-2.0, 2.0]` (honoured by vLLM) |
| `seed` | `int` | `null` | Random seed for reproducibility |
| `user` | `string` | `null` | Caller identifier, passed through for logging |

#### Non-streaming request

```bash
curl http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Explain gradient descent in one paragraph."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**Response**

```json
{
  "id": "chatcmpl-a3f7d...",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Gradient descent is an iterative optimisation algorithm..."
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 118,
    "total_tokens": 160
  }
}
```

#### Streaming request (SSE)

```bash
curl http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "stream": true
  }'
```

The response body is a stream of `data:` lines:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":", 2"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Python — `openai` SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:PORT/v1",
    api_key="not-used",          # Bhaskera's Ray endpoint has no auth
)

# Non-streaming
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=128,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Tell me a short joke."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Gateway API (with Authentication)

When `serve.gateway.enabled: true` in the config, Bhaskera starts a second
Uvicorn process (the **Langfuse Gateway**) on a separate port. This layer sits
in front of the Ray Serve deployment and adds:

- **API key authentication** — only pre-approved keys are accepted.
- **Per-user tracing** — each request is tagged with a user ID in Langfuse.
- **Cloudflare tunnel** — a public HTTPS URL via `./cloudflared`.

### Pre-configured API keys

The gateway ships with three keys for development:

| API Key | User ID |
|---|---|
| `sk-bhaskera-admin` | `admin` |
| `sk-bhaskera-alice` | `user_alice` |
| `sk-bhaskera-bob` | `user_bob` |

> **Production note:** Edit `src/bhaskera/gateway.py` → `VALID_KEYS` dict to add
> or rotate keys before deploying publicly.

### Gateway endpoint

The gateway exposes a single endpoint at the same path as the Ray backend:

```
POST /v1/chat/completions
```

All request and response fields are identical to the direct Ray endpoint —
the gateway is a transparent proxy with auth enforcement.

### Gateway request — curl

```bash
# Replace GATEWAY_PORT with the port printed in startup logs
# Replace URL with the Cloudflare public URL for remote access

curl https://random-phrase.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-bhaskera-alice" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [{"role": "user", "content": "What is FSDP2?"}],
    "temperature": 0.6,
    "max_tokens": 512,
    "stream": false
  }'
```

### Gateway request — Python

```python
from openai import OpenAI

# Via Cloudflare tunnel (public access)
client = OpenAI(
    base_url="https://random-phrase.trycloudflare.com/v1",
    api_key="sk-bhaskera-alice",
)

# Or via localhost (if running on the same machine as the server)
client = OpenAI(
    base_url=f"http://127.0.0.1:{GATEWAY_PORT}/v1",
    api_key="sk-bhaskera-admin",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "system", "content": "You are a medical assistant."},
        {"role": "user",   "content": "What are the symptoms of Type 2 diabetes?"},
    ],
    temperature=0.4,
    max_tokens=300,
)
print(response.choices[0].message.content)
```

### Gateway streaming — Python

```python
stream = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

### Invalid key response

```json
{
  "detail": "Invalid API Key"
}
```
HTTP status: `401 Unauthorized`

---

## Backends

### vLLM (recommended for GPU)

Selected via `serve.backend: vllm` in the config.

- Uses `vllm.AsyncLLMEngine` — continuous batching, PagedAttention, tensor parallelism.
- All concurrency is managed by vLLM internally; Ray Serve places no cap on requests per replica.
- Tensor parallelism is configured via `serve.vllm.tensor_parallel_size`.
- For a 17B model on a 40 GB GPU, set `tensor_parallel_size: 2` and `ray_actor_options.num_gpus: 2`.

### HuggingFace (fallback)

Selected via `serve.backend: hf`.

- Uses the same `build_model` path as training — fully supports LoRA, trust_remote_code
  models, and custom architectures that vLLM does not.
- `model.generate()` is not thread-safe; Ray Serve serialises requests per replica
  (`max_ongoing_requests: 1`). Scale horizontally with `serve.num_replicas` instead.
- Streaming is implemented via `TextIteratorStreamer` dispatched to a background
  thread, keeping the FastAPI event loop unblocked between tokens.

---

## Autoscaling

Enable autoscaling by setting both bounds in the config:

```yaml
serve:
  autoscaling_min_replicas: 1
  autoscaling_max_replicas: 4
```

Ray Serve will scale between these bounds based on pending request count.
`num_replicas` is ignored when autoscaling is active.

---

## Module Layout

```
src/bhaskera/serve/
├── __init__.py       # public re-exports
├── app.py            # build_app() — translates Config → Ray Serve Application
├── deployment.py     # LLMDeployment — Ray Serve actor, FastAPI routes
├── engine.py         # BaseEngine, VLLMEngineWrapper, HFEngineWrapper, create_engine()
└── schemas.py        # Pydantic v2 models mirroring the OpenAI spec
```

The gateway lives one level up at `src/bhaskera/gateway.py` and is launched as a
subprocess by the serve launcher when `serve.gateway.enabled: true`.

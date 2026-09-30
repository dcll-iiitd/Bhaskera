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

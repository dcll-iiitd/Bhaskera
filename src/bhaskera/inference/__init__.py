"""
bhaskera.inference
==================
Inference capability for the Bhaskera LLM framework.

Backends
--------
``RayVLLMBackend``  — Ray actor wrapping vLLM (primary, CUDA)
``HFBackend``       — HuggingFace generate() (fallback / CPU / dev)

Quick start
-----------
    from bhaskera.config import load_config
    from bhaskera.inference import InferenceEngine

    cfg    = load_config("configs/inference_param2.yaml")
    engine = InferenceEngine(cfg)
    texts  = engine.generate(["Tell me about quantum computing"])
    print(texts[0])

    # Param2-Thinking structured output
    engine = InferenceEngine.from_param2()
    for out in engine.generate_param2(["What is 2+2?"]):
        print(out.reasoning[:100])
        print(out.final_answer)

Config YAML keys (under ``inference:``)
----------------------------------------
    max_new_tokens: 512
    temperature:    0.7
    top_p:          0.9
    top_k:          50
    do_sample:      true
    kv_cache:       static        # static | turboquant | none  (turboquant = HF path only)
    device:         auto          # cuda | cpu | mps | auto

    # vLLM-specific (Ray/vLLM backend)
    max_model_len:            4096
    gpu_memory_utilization:   0.95
    enforce_eager:            false
    enable_prefix_caching:    true
    tensor_parallel_size:     1    # override per CLI --tensor-parallel-size

    # TurboQuant (HF backend only for now)
    turboquant:
      key_bits:          4
      value_bits:        2
      residual_window:   128
      protected_layers:  2

    # Speculative decoding (HF backend only)
    speculative:
      enabled:          false
      draft_model_name: ""
      num_draft_tokens: 5
"""
from .engine import InferenceEngine

# KV cache utilities (used by HF backend and tests)
from .kv_cache import (
    BaseKVCache,
    StaticKVCache,
    TurboQuantKVCache,
    build_kv_cache,
)

# Quantizer
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max

# Sampling
from .sampling import (
    greedy_sample,
    sample_from_logits,
    temperature_scale,
    top_k_filter,
    top_p_filter,
)

# Speculative decoding (HF path)
from .speculative import SpeculativeDecoder, build_speculative_decoder

__all__ = [
    # Engine
    "InferenceEngine",
    # Backends (re-exported for programmatic use)
    "RayVLLMBackend",
    "HFBackend",
    # KV caches
    "BaseKVCache",
    "StaticKVCache",
    "TurboQuantKVCache",
    "build_kv_cache",
    # Quantizer
    "LloydMaxCodebook",
    "solve_lloyd_max",
    # Sampling
    "greedy_sample",
    "sample_from_logits",
    "temperature_scale",
    "top_k_filter",
    "top_p_filter",
    # Speculative
    "SpeculativeDecoder",
    "build_speculative_decoder",
]

# Lazy backend imports so Ray/vLLM are not required at package import time
def __getattr__(name: str):
    if name == "RayVLLMBackend":
        from .backends.ray_vllm import RayVLLMBackend
        return RayVLLMBackend
    if name == "HFBackend":
        from .backends.hf import HFBackend
        return HFBackend
    raise AttributeError(f"module 'bhaskera.inference' has no attribute {name!r}")
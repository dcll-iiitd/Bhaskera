"""
bhaskera.inference.backends
============================
Pluggable inference backends.

  RayVLLMBackend  — Ray actor wrapping vLLM (primary, CUDA only)
  HFBackend       — HuggingFace generate() (fallback / CPU / dev)
"""
from .ray_vllm import RayVLLMBackend
from .hf import HFBackend

__all__ = ["RayVLLMBackend", "HFBackend"]
"""
bhaskera.inference.engine
=========================
Public ``InferenceEngine`` facade.

Backend selection priority
--------------------------
1. ``RayVLLMBackend``   — Ray actor wrapping vLLM (best throughput, any scale)
2. ``HFBackend``        — HuggingFace generate() (fallback / no-GPU dev mode)

Force a specific backend via the environment variable::

    BHASKERA_BACKEND=hf     bhaskera-infer ...
    BHASKERA_BACKEND=vllm   bhaskera-infer ...    (default when vLLM present)

TurboQuant
----------
KV-cache quantisation is not yet integrated into the Ray/vLLM path.
The config keys are preserved for forward compatibility; passing
``kv_cache: turboquant`` currently logs a warning and falls back to
vLLM's native paged KV management (which is already very memory-efficient).
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "auto":     None,
}


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Unified inference facade for Bhaskera.

    Usage (simple)::

        from bhaskera.config import load_config
        from bhaskera.inference import InferenceEngine

        cfg    = load_config("configs/inference_param2.yaml")
        engine = InferenceEngine(cfg)
        engine.load()
        texts  = engine.generate(["Explain attention mechanisms."])
        print(texts[0])

    Usage (Param2 structured output)::

        engine = InferenceEngine.from_param2()
        for out in engine.generate_param2(["What is 2+2?"]):
            print(out.reasoning[:100])
            print(out.final_answer)
    """

    def __init__(self, cfg, model_name: Optional[str] = None):
        self.cfg        = cfg
        self.infer_cfg  = cfg.inference
        self.model_name = model_name or cfg.model.name
        self.device     = _resolve_device(self.infer_cfg.device)
        self._backend   = None
        self._backend_name: str = "not_loaded"
        self._loaded    = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "InferenceEngine":
        if self._loaded:
            return self

        # Warn if turboquant requested (not yet implemented in Ray/vLLM path)
        if getattr(self.infer_cfg, "kv_cache", "static") == "turboquant":
            logger.warning(
                "kv_cache=turboquant is not yet integrated with the Ray/vLLM "
                "backend.  vLLM's native paged-attention memory management will "
                "be used instead (still very efficient).  TurboQuant integration "
                "is planned for a future release."
            )

        env_override = os.environ.get("BHASKERA_BACKEND", "").strip().lower()

        if env_override == "hf":
            self._load_hf()
        elif env_override == "vllm":
            self._load_vllm()
        else:
            # Auto: prefer Ray+vLLM on CUDA, fall back to HF
            if self.device.type == "cuda" and _vllm_available():
                try:
                    self._load_vllm()
                except Exception as exc:
                    logger.warning(
                        f"Ray/vLLM backend failed ({exc}), falling back to HF backend"
                    )
                    self._load_hf()
            else:
                self._load_hf()

        self._loaded = True
        return self

    def _load_vllm(self) -> None:
        from bhaskera.inference.backends.ray_vllm import RayVLLMBackend
        self._backend = RayVLLMBackend(self.model_name, self.cfg, self.device)
        self._backend_name = "ray_vllm"
        logger.info(f"[Engine] Using Ray+vLLM backend for {self.model_name}")

    def _load_hf(self) -> None:
        from bhaskera.inference.backends.hf import HFBackend
        self._backend = HFBackend(self.model_name, self.cfg, self.device)
        self._backend_name = "hf"
        logger.info(f"[Engine] Using HF backend for {self.model_name}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        return_full_text: bool = False,
    ) -> List[str]:
        """
        Generate text for one or more prompts.

        Returns a list of generated strings (one per prompt).
        If ``return_full_text=True`` the prompt is prepended to each output.
        """
        if not self._loaded:
            self.load()

        if isinstance(prompts, str):
            prompts = [prompts]

        g = self.infer_cfg
        return self._backend.generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else g.max_new_tokens,
            temperature=temperature    if temperature    is not None else g.temperature,
            top_p=top_p               if top_p          is not None else g.top_p,
            top_k=top_k               if top_k          is not None else g.top_k,
            do_sample=do_sample       if do_sample       is not None else g.do_sample,
            return_full_text=return_full_text,
        )

    # ------------------------------------------------------------------
    # Param2 structured interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_param2(
        self,
        prompts: Union[str, List[str]],
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> list:
        """
        Param2-Thinking structured generation.

        Applies the chat template, decodes with ``skip_special_tokens=False``
        (required so ``<think>`` tags survive), and returns a list of
        :class:`~bhaskera.inference.param2.Param2Output` objects with
        ``reasoning``, ``tool_calls``, and ``final_answer`` fields.
        """
        from bhaskera.inference.param2 import parse_model_output

        if not self._loaded:
            self.load()
        if isinstance(prompts, str):
            prompts = [prompts]

        g = self.infer_cfg
        raw_texts = self._backend.generate_param2(
            prompts=prompts,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else g.max_new_tokens,
            temperature=temperature       if temperature    is not None else g.temperature,
            top_p=top_p                   if top_p          is not None else g.top_p,
            top_k=top_k                   if top_k          is not None else g.top_k,
            do_sample=do_sample           if do_sample       is not None else g.do_sample,
        )
        return [parse_model_output(t) for t in raw_texts]

    # ------------------------------------------------------------------
    # Stats / housekeeping
    # ------------------------------------------------------------------

    def kv_cache_stats(self) -> Optional[dict]:
        """Return KV-cache compression stats (available when TurboQuant active)."""
        if not self._loaded:
            return None
        if hasattr(self._backend, "kv_cache_stats"):
            return self._backend.kv_cache_stats()
        return None

    def shutdown(self) -> None:
        """Release Ray actors / model memory."""
        if hasattr(self._backend, "shutdown"):
            self._backend.shutdown()
        self._loaded = False
        self._backend = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_param2(
        cls,
        model_name: str = "bharatgenai/Param2-17B-A2.4B-Thinking",
        kv_cache: str = "static",
        device: str = "auto",
        **generation_overrides,
    ) -> "InferenceEngine":
        """
        Convenience constructor pre-configured for Param2-17B-A2.4B-Thinking.

        Example::

            engine = InferenceEngine.from_param2()
            outputs = engine.generate_param2(["Explain quantum entanglement."])
            print(outputs[0].final_answer)
        """
        from bhaskera.inference.param2 import build_param2_config
        cfg = build_param2_config(
            model_name=model_name,
            kv_cache=kv_cache,
            device=device,
            **generation_overrides,
        )
        engine = cls(cfg)
        engine.load()
        return engine

    def __repr__(self) -> str:
        return (
            f"InferenceEngine(model={self.model_name!r}, "
            f"device={self.device}, backend={self._backend_name!r})"
        )
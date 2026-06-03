"""
bhaskera.serve.engine
=====================
Async engine abstraction with two concrete backends.

VLLMEngineWrapper  (fast path)
    Uses ``vllm.AsyncLLMEngine`` for continuous batching and tensor
    parallelism on any vLLM-supported architecture.

HFEngineWrapper  (fallback)
    Uses ``bhaskera.models.loader.build_model`` — the same model-loading
    path as training — so custom/research architectures that vLLM doesn't
    support (MoEs, trust_remote_code models, LoRA-patched networks) work
    transparently.

    Because ``model.generate()`` is blocking, every call is dispatched via
    ``asyncio.to_thread()`` so the FastAPI event loop is never frozen.
    Per-token streaming is achieved with HF's ``TextIteratorStreamer``;
    each ``next()`` call is individually dispatched to the thread pool.

Factory::

    engine = create_engine(cfg)   # returns VLLMEngineWrapper or HFEngineWrapper
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, Optional
import os
import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from bhaskera.config import Config

logger = logging.getLogger(__name__)

# Sentinel value used to signal end of a generator without relying on
# StopIteration propagation through asyncio coroutines.
_SENTINEL = object()
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_FLASHINFER"] = "0"

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

@dataclass
class GenerationParams:
    """
    Backend-agnostic generation hyperparameters.
    The deployment layer maps ChatCompletionRequest fields to this object.
    """
    max_new_tokens:  int        = 512
    temperature:     float      = 1.0
    top_p:           float      = 0.9
    top_k:           int        = 50
    stop_sequences:  list[str]  = field(default_factory=list)
    seed:            Optional[int] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEngine(ABC):
    """
    Contract for all LLM backends.

    ``get_tokenizer()``
        Returns the HF tokenizer used by the deployment layer to format
        messages into a prompt via ``apply_chat_template``.

    ``generate(prompt, params)``
        Async generator — must be implemented by subclasses using ``yield``.
        Yields decoded text chunks (strings) incrementally.

    ``generate_full(prompt, params)``
        Convenience wrapper: collects all chunks into a single string.
        Override if the backend has a more efficient non-streaming path.
    """

    @abstractmethod
    def get_tokenizer(self):
        """Return the HuggingFace PreTrainedTokenizer for this engine."""
        ...

    async def generate(
        self,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields decoded text chunks.
        Subclasses MUST implement this as an ``async def … yield`` function.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override generate()."
        )
        yield  # pragma: no cover — makes the ABC itself an async generator

    async def generate_full(self, prompt: str, params: GenerationParams) -> str:
        """Collect all streaming chunks into one string."""
        return "".join([chunk async for chunk in self.generate(prompt, params)])


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

class VLLMEngineWrapper(BaseEngine):
    """
    High-throughput backend.

    ``vllm`` is an *optional* dependency (``pip install 'bhaskera[vllm]'``).
    The import is deferred to ``__init__`` so the module loads cleanly even
    when vLLM is not installed — the ImportError surfaces only when the
    vLLM backend is actually requested.

    vLLM handles its own tokenization internally; we load a separate HF
    tokenizer *only* for ``apply_chat_template`` in the deployment layer.
    """

    def __init__(self, cfg: "Config") -> None:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed.  "
                "Run: pip install 'bhaskera[serve,vllm]' or pip install vllm>=0.4.3"
            ) from exc

        vc = cfg.serve.vllm
        engine_args = AsyncEngineArgs(
            model=cfg.model.name,
            tensor_parallel_size=vc.tensor_parallel_size,
            gpu_memory_utilization=vc.gpu_memory_utilization,
            max_model_len=vc.max_model_len,
            dtype=vc.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
            enforce_eager=vc.enforce_eager,
            attention_backend="TRITON_ATTN",
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Chat-template tokenizer (vLLM loads its own internals for generation).
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        logger.info(
            "VLLMEngineWrapper ready | model=%s tp=%d",
            cfg.model.name, vc.tensor_parallel_size,
        )

    def get_tokenizer(self):
        return self._tokenizer

    async def generate(
        self,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            max_tokens=params.max_new_tokens,
            stop=params.stop_sequences or [],
            seed=params.seed,
        )

        request_id = f"bhaskera-{uuid.uuid4().hex}"
        prev_len = 0

        # vLLM yields RequestOutput objects where outputs[0].text accumulates
        # the full generated string so far.  We diff to get the incremental delta.
        async for request_output in self._engine.generate(
            prompt, sampling_params, request_id
        ):
            if not request_output.outputs:
                continue
            cur_text = request_output.outputs[0].text
            delta = cur_text[prev_len:]
            if delta:
                yield delta
            prev_len = len(cur_text)


# ---------------------------------------------------------------------------
# HF fallback backend
# ---------------------------------------------------------------------------

class HFEngineWrapper(BaseEngine):
    """
    Fallback for custom/research architectures not supported by vLLM.

    Uses ``bhaskera.models.loader.build_model`` so LoRA, Liger Kernel
    patching, and trust_remote_code models all work identically to training.

    Concurrency note
    ----------------
    ``torch.nn.Module.generate()`` is not thread-safe.  This wrapper is
    deployed with ``max_concurrent_queries=1`` (set in ``serve/app.py``)
    so Ray Serve serialises requests per replica.  Scale horizontally with
    ``cfg.serve.num_replicas`` instead.

    Streaming implementation
    ------------------------
    ``TextIteratorStreamer`` bridges the blocking generation thread to the
    async consumer (FastAPI response handler):

    1. ``model.generate()`` runs in a daemon thread; it puts decoded tokens
       into the streamer's internal queue as they are produced.
    2. ``_drain_streamer()`` awaits each ``next()`` call via
       ``asyncio.to_thread()`` — each call blocks briefly in the executor
       thread while the generation thread produces the next token, but the
       asyncio event loop is released between tokens.
    """

    def __init__(self, cfg: "Config") -> None:
        from bhaskera.models.loader import build_model

        hc = cfg.serve.hf
        device_str = hc.device
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        logger.info(
            "HFEngineWrapper: loading %s on %s …", cfg.model.name, device
        )
        self._model, self._profile = build_model(cfg, device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        # Ensure a pad token exists (some models have only eos).
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._device = device
        logger.info(
            "HFEngineWrapper ready | model=%s device=%s moe=%s",
            cfg.model.name, device, self._profile.is_moe,
        )

    def get_tokenizer(self):
        return self._tokenizer

    # ------------------------------------------------------------------
    # Stopping criteria helper
    # ------------------------------------------------------------------

    def _build_stopping_criteria(
        self,
        stop_sequences: list[str],
        prompt_len: int,
    ):
        """
        Construct a ``StoppingCriteriaList`` for the given stop strings.
        Returns ``None`` when the list is empty (avoids extra overhead).

        Token-level check: generation halts as soon as the suffix of the
        output token IDs matches any stop sequence encoding.
        """
        from transformers import StoppingCriteria, StoppingCriteriaList

        if not stop_sequences:
            return None

        class _StopOnStrings(StoppingCriteria):
            def __init__(self_, tokenizer, stops: list[str], plen: int):
                self_.stop_ids_list = [
                    tokenizer.encode(s, add_special_tokens=False)
                    for s in stops
                ]
                self_.prompt_len = plen

            def __call__(
                self_,
                input_ids: torch.Tensor,
                scores: torch.Tensor,
                **kwargs,
            ) -> bool:
                generated = input_ids[0][self_.prompt_len:].tolist()
                for stop_ids in self_.stop_ids_list:
                    n = len(stop_ids)
                    if n and generated[-n:] == stop_ids:
                        return True
                return False

        return StoppingCriteriaList(
            [_StopOnStrings(self._tokenizer, stop_sequences, prompt_len)]
        )

    # ------------------------------------------------------------------
    # Async streaming generate
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncGenerator[str, None]:
        from transformers import TextIteratorStreamer

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_len: int = inputs["input_ids"].shape[1]

        # ``timeout=None`` means the streamer blocks indefinitely until the
        # next token is ready, which is what we want — we rely on the
        # generation thread to terminate naturally via eos / max_new_tokens.
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
        )

        stopping_criteria = self._build_stopping_criteria(
            params.stop_sequences, prompt_len
        )

        gen_kwargs: dict = {
            **inputs,
            "streamer":       streamer,
            "max_new_tokens": params.max_new_tokens,
            "do_sample":      params.temperature > 0.0,
            # Avoid division-by-zero when temperature == 0 (greedy).
            "temperature":    max(params.temperature, 1e-6),
            "top_p":          params.top_p,
            "top_k":          params.top_k,
            "pad_token_id":   self._tokenizer.pad_token_id,
            "eos_token_id":   self._tokenizer.eos_token_id,
        }
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria
        if params.seed is not None:
            torch.manual_seed(params.seed)

        # Kick off generation in a background daemon thread.
        # The thread writes tokens into the streamer's internal queue;
        # we read them out below via asyncio.to_thread.
        threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
            name="bhaskera-hf-generate",
        ).start()

        # Drain the streamer one token at a time.
        # Each `next(streamer)` call blocks briefly while the generate
        # thread produces the next token.  asyncio.to_thread releases the
        # event loop during that wait, keeping the server responsive.
        def _safe_next() -> object:
            try:
                return next(streamer)
            except StopIteration:
                return _SENTINEL

        while True:
            token: object = await asyncio.to_thread(_safe_next)
            if token is _SENTINEL:
                break
            # TextIteratorStreamer may emit empty strings during flushing.
            if token:
                yield token  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_engine(cfg: "Config") -> BaseEngine:
    """
    Return the appropriate engine based on ``cfg.serve.backend``.

    "vllm"  →  VLLMEngineWrapper  (requires pip install vllm>=0.4.3)
    "hf"    →  HFEngineWrapper    (requires pip install bhaskera[serve])
    """
    backend = cfg.serve.backend.lower().strip()
    if backend == "vllm":
        return VLLMEngineWrapper(cfg)
    if backend == "hf":
        return HFEngineWrapper(cfg)
    raise ValueError(
        f"Unknown serve backend {backend!r}.  "
        f"Set cfg.serve.backend to 'vllm' or 'hf'."
    )

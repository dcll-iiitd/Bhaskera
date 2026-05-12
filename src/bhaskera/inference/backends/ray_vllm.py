"""
bhaskera.inference.backends.ray_vllm
=====================================
Ray-orchestrated vLLM backend.

Architecture
------------
The vLLM ``LLM`` object owns CUDA memory for its entire lifetime.  To keep
the driver process lean (and to support multi-node SLURM jobs where the
head node has no GPU), the engine runs inside a **Ray actor** on a worker
that has GPUs.

Topology
~~~~~~~~
::

    Driver process (CPU)
        │
        │  ray.remote call
        ▼
    _VLLMWorkerActor  (Ray actor, 1+ GPUs)
        │
        │  vLLM LLM.generate()
        ▼
    vLLM engine  (PagedAttention, CUDA graphs, continuous batching)

The actor is created once per ``RayVLLMBackend`` instance and lives until
``shutdown()`` is called or the driver exits.

Ray init strategy
~~~~~~~~~~~~~~~~~
1. If ``RAY_ADDRESS`` env var is set → join that cluster (SLURM head node).
2. Otherwise → ``ray.init()`` locally.
3. Already initialised → reuse (safe in training driver that called
   ``ray.init()`` itself).

SLURM
~~~~~
Set ``RAY_ADDRESS=<head_node_ip>:<port>`` in your SLURM batch script before
launching.  The ``bhaskera-infer`` CLI accepts ``--ray-address`` which sets
this env var for you.

Multi-GPU / tensor-parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``tensor_parallel_size`` defaults to ``torch.cuda.device_count()`` when
>1 GPU is visible.  Set ``--tensor-parallel-size 1`` to force single-GPU.

TurboQuant (future)
~~~~~~~~~~~~~~~~~~~
The ``tq_cfg`` parameter is accepted but not yet wired.  vLLM's native
paged-attention already manages KV memory efficiently.  TurboQuant
integration will be added in a future milestone via vLLM's
``kv_cache_dtype`` hook.
"""
from __future__ import annotations

import inspect
import logging
import os
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ray + vLLM actor
# ---------------------------------------------------------------------------

def _build_vllm_actor_cls():
    """
    Construct the Ray remote actor class at call time so that importing
    this module never fails when Ray is absent (the class definition itself
    calls ray.remote which requires ray to be importable).
    """
    import ray

    @ray.remote
    class _VLLMWorkerActor:
        """
        Ray actor that owns the vLLM LLM engine.

        One actor is created per ``RayVLLMBackend`` instance.  The actor
        pins to ``num_gpus`` GPUs; vLLM handles tensor-parallelism internally.
        """

        def __init__(
            self,
            model_name: str,
            dtype: str,
            tensor_parallel_size: int,
            max_model_len: int,
            gpu_memory_utilization: float,
            trust_remote_code: bool,
            enforce_eager: bool,
            enable_prefix_caching: bool,
        ):
            from vllm import LLM  # type: ignore

            logger.info(
                f"[VLLMWorker] Loading {model_name!r} "
                f"(tp={tensor_parallel_size}, dtype={dtype})"
            )
            t0 = time.perf_counter()
            self._llm = LLM(
                model=model_name,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
                enable_prefix_caching=enable_prefix_caching,
            )
            elapsed = time.perf_counter() - t0
            logger.info(f"[VLLMWorker] Engine ready in {elapsed:.1f}s")

        # ------------------------------------------------------------------
        # Core generation (mirrors the example script exactly)
        # ------------------------------------------------------------------

        def generate(
            self,
            prompts: List[str],
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            do_sample: bool,
        ) -> List[dict]:
            """
            Run vLLM generation and return structured result dicts so the
            driver can compute token/s metrics without a GPU.

            Returns list of::

                {
                    "text":             str,
                    "generated_tokens": int,
                    "prompt_tokens":    int,
                }
            """
            from vllm import SamplingParams  # type: ignore

            params = SamplingParams(
                temperature=temperature if do_sample else 0.0,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                max_tokens=max_new_tokens,
            )

            outputs = self._llm.generate(prompts, params)

            results = []
            for out in outputs:
                results.append({
                    "text":             out.outputs[0].text,
                    "generated_tokens": len(out.outputs[0].token_ids),
                    "prompt_tokens":    len(out.prompt_token_ids),
                })
            return results

        # ------------------------------------------------------------------
        # Param2-Thinking: apply chat template inside actor (tokenizer lives here)
        # ------------------------------------------------------------------

        def generate_with_chat_template(
            self,
            user_messages: List[str],
            system_prompt: str,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            do_sample: bool,
        ) -> List[dict]:
            """
            Apply Param2's chat template then generate.  The tokenizer is
            available inside vLLM's engine; we reconstruct it here for
            template application.
            """
            from vllm import SamplingParams  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            model_name = self._llm.llm_engine.model_config.model

            # Load tokenizer (cached after first call in this actor process)
            if not hasattr(self, "_tok"):
                self._tok = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )

            # Build prompts with chat template
            formatted = []
            for user_msg in user_messages:
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ]
                text = self._tok.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted.append(text)

            params = SamplingParams(
                temperature=temperature if do_sample else 0.0,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                max_tokens=max_new_tokens,
                # skip_special_tokens=False so <think> tags survive
                skip_special_tokens=False,
            )

            outputs = self._llm.generate(formatted, params)

            results = []
            for out in outputs:
                results.append({
                    "text":             out.outputs[0].text,
                    "generated_tokens": len(out.outputs[0].token_ids),
                    "prompt_tokens":    len(out.prompt_token_ids),
                })
            return results

        def ping(self) -> str:
            """Health check."""
            return "ok"

    return _VLLMWorkerActor


# ---------------------------------------------------------------------------
# Ray init helper
# ---------------------------------------------------------------------------

def _ensure_ray_initialized(num_gpus: int) -> None:
    """
    Connect to or start a Ray cluster.

    Priority:
      1. RAY_ADDRESS env var → join existing cluster (SLURM use case)
      2. Already initialised  → no-op
      3. Otherwise            → local init
    """
    import ray

    if ray.is_initialized():
        logger.debug("[Ray] Already initialised — reusing session")
        return

    address = os.environ.get("RAY_ADDRESS", "").strip()
    if address:
        logger.info(f"[Ray] Connecting to cluster at {address}")
        ray.init(address=address, ignore_reinit_error=True)
    else:
        logger.info(f"[Ray] Starting local session ({num_gpus} GPU(s))")
        ray.init(
            num_gpus=num_gpus,
            num_cpus=os.cpu_count() or 4,
            ignore_reinit_error=True,
            # Suppress Ray's noisy startup banner in library use
            log_to_driver=True,
        )

    logger.info(f"[Ray] Resources: {ray.available_resources()}")


# ---------------------------------------------------------------------------
# Public backend class
# ---------------------------------------------------------------------------

class RayVLLMBackend:
    """
    Inference backend that runs vLLM inside a Ray actor.

    Parameters
    ----------
    model_name : str
        HuggingFace model id or local path.
    cfg : Config
        Bhaskera config object (reads ``cfg.model`` and ``cfg.inference``).
    device : torch.device
        Resolved device.  Must be CUDA.
    tensor_parallel_size : int | None
        Number of GPUs for tensor parallelism.  ``None`` → auto-detect.
    """

    def __init__(
        self,
        model_name: str,
        cfg,
        device: torch.device,
        tensor_parallel_size: Optional[int] = None,
    ):
        if device.type != "cuda":
            raise RuntimeError(
                "RayVLLMBackend requires CUDA.  "
                "Set BHASKERA_BACKEND=hf for CPU inference."
            )

        self._model_name = model_name
        self._cfg        = cfg
        self._device     = device

        infer_cfg = cfg.inference
        model_cfg = cfg.model

        # Resolve tensor-parallel size
        n_gpus = tensor_parallel_size or torch.cuda.device_count() or 1
        # Clamp to 1 for single-GPU setups
        if n_gpus == 0:
            n_gpus = 1

        raw_dtype = getattr(model_cfg, "dtype", "bfloat16")
        dtype_str = "auto" if raw_dtype == "auto" else raw_dtype

        # Ensure Ray is up before creating actor
        _ensure_ray_initialized(n_gpus)

        # Build the actor class (deferred so Ray import is not required at
        # module import time)
        ActorCls = _build_vllm_actor_cls()

        logger.info(
            f"[RayVLLM] Creating worker actor: model={model_name!r} "
            f"tp={n_gpus} dtype={dtype_str}"
        )

        # Request n_gpus GPUs for the actor
        self._actor = ActorCls.options(num_gpus=n_gpus).remote(
            model_name=model_name,
            dtype=dtype_str,
            tensor_parallel_size=n_gpus,
            max_model_len=getattr(infer_cfg, "max_model_len", 4096),
            gpu_memory_utilization=getattr(infer_cfg, "gpu_memory_utilization", 0.95),
            trust_remote_code=getattr(model_cfg, "trust_remote_code", False),
            # enforce_eager=False lets vLLM use CUDA graphs (faster decode)
            enforce_eager=getattr(infer_cfg, "enforce_eager", False),
            enable_prefix_caching=getattr(infer_cfg, "enable_prefix_caching", True),
        )

        # Block until actor is healthy
        import ray
        ray.get(self._actor.ping.remote())
        logger.info("[RayVLLM] Worker actor ready ✓")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        return_full_text: bool = False,
    ) -> List[str]:
        import ray

        t0 = time.perf_counter()
        results: List[dict] = ray.get(
            self._actor.generate.remote(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
        )
        elapsed = time.perf_counter() - t0

        # Log aggregate throughput on the driver side
        total_gen  = sum(r["generated_tokens"] for r in results)
        total_prom = sum(r["prompt_tokens"]    for r in results)
        logger.info(
            f"[RayVLLM] {len(prompts)} prompt(s) | "
            f"prompt_tokens={total_prom} generated_tokens={total_gen} | "
            f"{elapsed:.3f}s | "
            f"{total_gen / elapsed:.1f} tok/s"
        )

        texts = [r["text"] for r in results]
        if return_full_text:
            texts = [p + t for p, t in zip(prompts, texts)]
        return texts

    def generate_param2(
        self,
        prompts: List[str],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> List[str]:
        """
        Generate with Param2 chat template applied inside the actor.
        Returns raw text strings (caller parses <think> tags).
        """
        import ray

        results: List[dict] = ray.get(
            self._actor.generate_with_chat_template.remote(
                user_messages=prompts,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
        )
        return [r["text"] for r in results]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def kv_cache_stats(self) -> Optional[dict]:
        # TurboQuant not yet integrated with Ray/vLLM path
        return None

    def shutdown(self) -> None:
        """Kill the Ray actor and release its GPU memory."""
        import ray
        if self._actor is not None:
            ray.kill(self._actor)
            self._actor = None
            logger.info("[RayVLLM] Worker actor killed")
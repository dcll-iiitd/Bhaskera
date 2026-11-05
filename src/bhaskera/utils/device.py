"""
bhaskera.utils.device
=====================
Single place that decides whether Bhaskera runs on CUDA GPUs or on CPU.

CPU mode is selected automatically when ``torch.cuda.is_available()`` is
False (laptops, CI), or forced with ``BHASKERA_FORCE_CPU=1`` (which also
hides GPUs from this process).  On GPU hosts nothing changes.

CPU mode applies these config overrides (see ``apply_cpu_overrides``):
    * training.distributed.strategy  -> "ddp"  (FSDP2 needs CUDA/NCCL)
    * model.dtype                    -> float32 (bf16/fp16 are slow or
                                        unsupported on most CPUs)
    * model.attn_impl                -> sdpa (flash-attn is CUDA-only)
    * model.quantization             -> none (bitsandbytes is CUDA-only)
    * model.use_liger_kernel         -> False (Triton / CUDA-only)
    * monitoring.metrics.cuda_memory -> False
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def force_cpu() -> bool:
    return os.environ.get("BHASKERA_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")


def use_cuda() -> bool:
    """True iff training should run on CUDA GPUs."""
    return (not force_cpu()) and torch.cuda.is_available()


def worker_device(local_rank: int) -> torch.device:
    """Device owned by a training worker."""
    return torch.device(f"cuda:{local_rank}") if use_cuda() else torch.device("cpu")


def apply_cpu_overrides(cfg) -> bool:
    """
    Mutate ``cfg`` in place so it is valid for CPU training.
    Idempotent; a no-op (returns False) when CUDA is in use.
    """
    if use_cuda():
        return False

    changes: list[str] = []

    def _set(obj, attr, value, label):
        if obj is not None and hasattr(obj, attr) and getattr(obj, attr) != value:
            changes.append(f"{label}: {getattr(obj, attr)!r} -> {value!r}")
            setattr(obj, attr, value)

    _set(getattr(cfg.training, "distributed", None), "strategy", "ddp", "training.distributed.strategy")
    _set(cfg.model, "dtype", "float32", "model.dtype")
    if getattr(cfg.model, "attn_impl", None) not in (None, "sdpa", "eager"):
        _set(cfg.model, "attn_impl", "sdpa", "model.attn_impl")
    _set(cfg.model, "quantization", "none", "model.quantization")
    _set(cfg.model, "use_liger_kernel", False, "model.use_liger_kernel")
    metrics = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    _set(metrics, "cuda_memory", False, "monitoring.metrics.cuda_memory")

    if changes:
        logger.warning("CPU mode (no CUDA): applied config overrides: " + "; ".join(changes))
    return True

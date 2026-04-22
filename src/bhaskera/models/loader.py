"""
bhaskera.models.loader
======================
Load an HF `AutoModelForCausalLM` (or a registered custom loader), run
introspection, and optionally attach LoRA.

Changes vs v1:
  * Removed `model_config.output_router_logits = True` mutation at load
    time.  The training loop now passes `output_router_logits=True` per
    forward call when the profile reports aux-loss support, so there is
    no duplicated state to keep in sync.
  * `torch_dtype="auto"` no longer resolves through our DTYPE_MAP; we
    hand it directly to HF and then read the true dtype off the loaded
    params via introspection.
"""
from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from bhaskera.introspect import ModelProfile, introspect_model

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}

_CUSTOM_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Register a custom non-HF model loader under `name`."""
    def _wrap(fn: Callable):
        _CUSTOM_REGISTRY[name] = fn
        return fn
    return _wrap


def build_model(cfg, device: torch.device) -> Tuple[torch.nn.Module, ModelProfile]:
    """
    Load model, introspect it, optionally apply LoRA.

    For FSDP2, callers should pass device=torch.device("cpu") — the FSDP
    wrap step will shard and migrate the params to the correct GPU.
    For DDP, pass the target CUDA device directly.
    """
    name = cfg.model.name
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)

    raw_dtype = getattr(cfg.model, "dtype", "bfloat16")
    if raw_dtype == "auto":
        load_dtype: "str | torch.dtype" = "auto"
    else:
        load_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    kwargs: dict = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=load_dtype,
    )
    if cfg.model.attn_impl:
        kwargs["attn_implementation"] = cfg.model.attn_impl

    # ── Load model ──────────────────────────────────────────────────
    if name in _CUSTOM_REGISTRY:
        model = _CUSTOM_REGISTRY[name](cfg, device)
    else:
        model_config = AutoConfig.from_pretrained(
            name, trust_remote_code=trust_remote_code
        )
        # NOTE: we deliberately do NOT set max_position_embeddings nor
        # output_router_logits here.  Truncation to seq_len is the
        # tokenizer's job, and router logits are requested per-forward
        # by the training loop when MoE aux loss is needed.
        model = AutoModelForCausalLM.from_pretrained(
            name, config=model_config, **kwargs
        )
        if device.type != "cpu":
            model = model.to(device)

    # ── Introspect (never hardcode layer classes) ───────────────────
    profile = introspect_model(model)
    if raw_dtype != "auto":
        profile.model_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded {name} | params={param_count:,} | "
        f"dtype={profile.model_dtype} | moe={profile.is_moe}"
    )

    # ── LoRA ────────────────────────────────────────────────────────
    if cfg.lora.enabled:
        from .lora import apply_lora
        model = apply_lora(model, cfg, profile)

    return model, profile
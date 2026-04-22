"""
bhaskera.models.lora
====================
LoRA application with MoE awareness.

DTYPE NOTE (important — do not remove the cast below):
    PEFT initialises LoRA A/B in float32 for numerical stability.  FSDP2's
    `_init_mp_dtypes()` requires every parameter inside a shard group to
    share the same ORIGINAL dtype — MixedPrecisionPolicy only controls
    forward-time casting, not shard-init uniformity.  If the base model
    is loaded in bf16, leaving LoRA params in fp32 raises:

        AssertionError: FSDP expects uniform original parameter dtype
        but got {torch.float32, torch.bfloat16}

    So we cast LoRA params (the only trainable ones) to match the base
    dtype at apply time.  This is the same pattern HF Trainer and PEFT's
    own FSDP examples use.

    Trade-off: AdamW moments for LoRA are stored in bf16 instead of fp32,
    which is slightly noisier.  In practice this is fine for LoRA SFT —
    the loss of precision is small and `reduce_dtype=float32` in the FSDP
    MixedPrecisionPolicy keeps gradient reductions in fp32.

    If you need true fp32 master weights for LoRA, load the BASE model in
    fp32 instead (set `model.dtype: float32` in your config) and let
    MixedPrecisionPolicy(param_dtype=bf16) cast at forward time.  That
    costs ~2x the base-param memory but everything is uniform fp32 at
    shard-init so no manual cast is needed.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


def apply_lora(model: nn.Module, cfg, profile: ModelProfile) -> nn.Module:
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("pip install peft  # required for LoRA") from e

    # ── Resolve target modules ──────────────────────────────────────
    configured_targets = cfg.lora.target_modules
    if configured_targets == ["auto"] or configured_targets == "auto":
        target_modules: Optional[list[str]] = (
            list(profile.lora_targets) if profile.lora_targets else None
        )
        if target_modules is None:
            logger.warning(
                "Auto LoRA targets: introspection found nothing, "
                "falling back to PEFT defaults"
            )
    else:
        target_modules = list(configured_targets)

    # ── MoE: optionally include expert FFN linears ──────────────────
    if profile.is_moe and getattr(cfg.lora, "include_experts", False):
        if target_modules and configured_targets != ["auto"]:
            expert_linears = _find_expert_linear_names(model, profile)
            for short_name in expert_linears:
                if short_name not in target_modules:
                    target_modules.append(short_name)
            logger.info(f"Added expert LoRA targets: {expert_linears}")

    # ── Build PEFT config ───────────────────────────────────────────
    peft_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
    )
    if target_modules is not None:
        peft_kwargs["target_modules"] = target_modules

    modules_to_save = getattr(cfg.lora, "modules_to_save", [])
    if modules_to_save:
        peft_kwargs["modules_to_save"] = list(modules_to_save)

    lora_cfg = PeftLoraConfig(**peft_kwargs)
    model = get_peft_model(model, lora_cfg)

    # ── Cast LoRA (trainable) params to match base dtype ────────────
    # REQUIRED for FSDP2 (see module docstring).  We only touch trainable
    # params — frozen base weights already have the right dtype.
    target_dtype = profile.model_dtype
    cast_count = 0
    for pname, param in model.named_parameters():
        if param.requires_grad and param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
            cast_count += 1
    if cast_count:
        logger.info(
            f"Cast {cast_count} LoRA parameter tensors to {target_dtype} "
            "(FSDP2 requires uniform param dtype within a shard group)"
        )

    # ── Freeze router / gate weights (critical for MoE stability) ───
    if profile.is_moe and getattr(cfg.lora, "freeze_router", True):
        _freeze_router_weights(model, profile)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied | targets={target_modules} | "
        f"trainable={trainable:,} ({100 * trainable / total:.2f}%)"
    )
    return model


def _freeze_router_weights(model: nn.Module, profile: ModelProfile) -> None:
    """Freeze all gate/router parameters to prevent MoE routing collapse."""
    frozen = 0
    router_kw = ("gate", "router", "switch", "gating")
    for pname, param in model.named_parameters():
        is_router = any(r in pname for r in profile.router_module_names)
        if not is_router:
            is_router = any(kw in pname.lower() for kw in router_kw)
        if is_router and param.requires_grad:
            param.requires_grad_(False)
            frozen += 1
    if frozen:
        logger.info(f"Froze {frozen} router/gate parameter tensors")


def _find_expert_linear_names(model: nn.Module, profile: ModelProfile) -> list[str]:
    if not profile.expert_modules:
        return []
    sample = profile.expert_modules[0]
    names: set[str] = set()
    for name, mod in sample.named_modules():
        if isinstance(mod, nn.Linear):
            names.add(name.split(".")[-1])
    return sorted(names)

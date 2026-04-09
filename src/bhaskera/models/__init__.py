"""
bhaskera.models
===============
Model loading + optional LoRA wrapping + introspection.
Works for any HuggingFace AutoModelForCausalLM — dense or MoE.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple
import logging

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from bhaskera.introspect import introspect_model, ModelProfile

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}

# Optional custom model registry (for non-HF architectures)
_CUSTOM_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Register a custom model loader."""
    def _wrap(fn):
        _CUSTOM_REGISTRY[name] = fn
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_model(
    cfg,
    device: torch.device,
) -> Tuple[torch.nn.Module, ModelProfile]:
    """
    Load model from HuggingFace (or custom registry), introspect, and optionally
    apply LoRA.

    Returns (model, profile) — profile is used by distributed/ and trainer/.

    FSDP2: always load on CPU — the FSDP2 wrapper shards onto GPUs.
    DDP:   load directly on GPU.
    """
    name = cfg.model.name
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)

    # ── Resolve dtype ───────────────────────────────────────────────
    # "auto" means: load in whatever dtype the model was saved in,
    # then read the actual dtype from the model after loading.
    raw_dtype = getattr(cfg.model, "dtype", "bfloat16")
    if raw_dtype == "auto":
        load_dtype = "auto"  # HF handles this natively
    else:
        load_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    # ── Build kwargs ────────────────────────────────────────────────
    kwargs: dict = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    # torch_dtype="auto" tells HF to use the checkpoint's native dtype
    kwargs["torch_dtype"] = load_dtype

    if cfg.model.attn_impl:
        kwargs["attn_implementation"] = cfg.model.attn_impl

    # ── Load model ──────────────────────────────────────────────────
    if name in _CUSTOM_REGISTRY:
        model = _CUSTOM_REGISTRY[name](cfg, device)
    else:
        model_config = AutoConfig.from_pretrained(
            name, trust_remote_code=trust_remote_code
        )
        # NOTE: we do NOT override max_position_embeddings.
        # Sequence length is controlled via tokenizer truncation only.
        # Mutating this corrupts RoPE frequencies in many architectures.

        # For MoE models: enable router logits output so we get aux_loss
        if hasattr(model_config, "output_router_logits"):
            model_config.output_router_logits = True

        model = AutoModelForCausalLM.from_pretrained(
            name, config=model_config, **kwargs
        )
        if device.type != "cpu":
            model = model.to(device)

    # ── Introspect (zero hardcoding) ────────────────────────────────
    profile = introspect_model(model)

    # If dtype was "auto", profile already has the resolved dtype.
    # If explicit, set it on the profile so downstream uses a consistent value.
    if raw_dtype != "auto":
        profile.model_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded {name} | params={param_count:,} | dtype={profile.model_dtype} | "
        f"moe={profile.is_moe}"
    )

    # ── LoRA ────────────────────────────────────────────────────────
    if cfg.lora.enabled:
        model = _apply_lora(model, cfg, profile)

    return model, profile


# ---------------------------------------------------------------------------
# LoRA — MoE-aware
# ---------------------------------------------------------------------------

def _apply_lora(
    model: torch.nn.Module,
    cfg,
    profile: ModelProfile,
) -> torch.nn.Module:
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError("pip install peft  # required for LoRA")

    # ── Resolve target modules ──────────────────────────────────────
    configured_targets = cfg.lora.target_modules
    if configured_targets == ["auto"] or configured_targets == "auto":
        if profile.lora_targets:
            target_modules = list(profile.lora_targets)
        else:
            # Fallback: let PEFT figure it out
            logger.warning(
                "Auto LoRA targets: introspection found nothing, "
                "falling back to PEFT defaults"
            )
            target_modules = None  # PEFT uses its own heuristic
    else:
        target_modules = list(configured_targets)

    # For MoE: optionally include expert FFN linear layers
    if profile.is_moe and getattr(cfg.lora, "include_experts", False):
        # Expert linears are already in the auto-detected list
        # (they're nn.Linear inside expert modules).
        # If user provided explicit targets, we add expert-specific ones.
        if target_modules and configured_targets != ["auto"]:
            expert_linears = _find_expert_linear_names(model, profile)
            for name in expert_linears:
                if name not in target_modules:
                    target_modules.append(name)
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

    lora_cfg = LoraConfig(**peft_kwargs)
    model = get_peft_model(model, lora_cfg)

    # ── Cast LoRA params to match base model dtype (required for FSDP) ──
    target_dtype = profile.model_dtype
    for pname, param in model.named_parameters():
        if param.requires_grad and param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)

    # ── Freeze router / gate weights (critical for MoE stability) ───
    if profile.is_moe and getattr(cfg.lora, "freeze_router", True):
        _freeze_router_weights(model, profile)

    # ── Log ──────────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied | targets={target_modules} | "
        f"trainable={trainable:,} ({100 * trainable / total:.2f}%)"
    )

    return model


def _freeze_router_weights(model: torch.nn.Module, profile: ModelProfile) -> None:
    """Freeze all gate/router parameters to prevent MoE routing collapse."""
    frozen_count = 0
    for pname, param in model.named_parameters():
        # Match against known router module names from introspection
        is_router = any(
            router_name in pname for router_name in profile.router_module_names
        )
        # Also catch by keyword in case introspection missed some
        if not is_router:
            pname_lower = pname.lower()
            is_router = any(
                kw in pname_lower for kw in ("gate", "router", "switch", "gating")
            )
        if is_router and param.requires_grad:
            param.requires_grad_(False)
            frozen_count += 1
    if frozen_count:
        logger.info(f"Froze {frozen_count} router/gate parameter tensors")


def _find_expert_linear_names(
    model: torch.nn.Module, profile: ModelProfile
) -> list[str]:
    """Find unique short names of nn.Linear inside expert modules."""
    if not profile.expert_modules:
        return []
    import torch.nn as nn
    sample = profile.expert_modules[0]
    names = set()
    for name, mod in sample.named_modules():
        if isinstance(mod, nn.Linear):
            names.add(name.split(".")[-1])
    return sorted(names)

"""
bhaskera.distributed
====================
FSDP2 (torch.distributed.fsdp v2) and DDP wrappers.

MoE-aware: auto-detects expert modules and applies per-expert sharding
for optimal memory/communication. Dense models work unchanged.

FSDP2 is the default. It uses the new fully_shard() API which is simpler,
composes better with torch.compile, and avoids the deprecated
FSDP1 FullStateDictConfig dance.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def wrap_model(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    """
    Wrap model for distributed training.
    strategy: 'fsdp' → FSDP2   |   'ddp' → DDP

    Uses ModelProfile for auto-detection of layer classes, MoE topology, etc.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before wrap_model()")

    strategy = cfg.training.distributed.strategy.lower()
    if strategy == "fsdp":
        return _wrap_fsdp2(model, cfg, local_rank, profile)
    elif strategy == "ddp":
        return _wrap_ddp(model, cfg, local_rank, profile)
    else:
        raise ValueError(f"Unknown distributed strategy: '{strategy}'. Choose 'fsdp' or 'ddp'.")


# ---------------------------------------------------------------------------
# FSDP2 — torch.distributed._composable.fsdp.fully_shard
# ---------------------------------------------------------------------------

def _wrap_fsdp2(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    """
    Wrap with FSDP2 (fully_shard API).

    Sharding order (inside-out — inner modules first, then outer, then root):
      1. If MoE: shard each expert module individually
      2. Shard each decoder layer
      3. Shard root model

    This ensures that for MoE, only activated experts' params are all-gathered
    during forward pass, instead of all experts at once.
    """
    try:
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    except ImportError:
        raise ImportError(
            "FSDP2 requires PyTorch >= 2.4. "
            "Either upgrade PyTorch or set strategy: ddp in your config."
        )

    fsdp_cfg = cfg.training.distributed.fsdp

    # ── Mixed precision policy ──────────────────────────────────────
    mp_policy = _build_mp_policy(fsdp_cfg, profile, MixedPrecisionPolicy)

    # ── Resolve decoder layer class ─────────────────────────────────
    # Priority: explicit config > auto-detected from profile
    decoder_cls = _resolve_decoder_cls(model, fsdp_cfg, profile)

    # ── STEP 1: Per-expert sharding (MoE only) ─────────────────────
    if profile.is_moe and fsdp_cfg.shard_experts_individually and profile.expert_modules:
        expert_count = 0
        for expert_module in profile.expert_modules:
            fully_shard(expert_module, mp_policy=mp_policy)
            expert_count += 1
        logger.info(
            f"FSDP2: per-expert sharding applied to {expert_count} expert modules "
            f"(class={profile.expert_module_cls.__name__ if profile.expert_module_cls else 'N/A'})"
        )

    # ── STEP 2: Per-decoder-layer sharding ──────────────────────────
    if decoder_cls is not None:
        layer_count = 0
        for module in model.modules():
            if isinstance(module, decoder_cls):
                fully_shard(module, mp_policy=mp_policy)
                layer_count += 1
        logger.info(
            f"FSDP2: per-layer sharding applied to {layer_count} "
            f"{decoder_cls.__name__} layers"
        )
    else:
        logger.warning(
            "No decoder layer class found — "
            "applying fully_shard to root model only. "
            "This is suboptimal for memory."
        )

    # ── STEP 3: Shard root model ────────────────────────────────────
    fully_shard(model, mp_policy=mp_policy)

    # ── Activation checkpointing ────────────────────────────────────
    if fsdp_cfg.activation_checkpointing:
        _apply_activation_checkpointing(model, profile, decoder_cls)

    logger.info(
        f"FSDP2 wrap complete | "
        f"param_dtype={mp_policy.param_dtype} | "
        f"reduce_dtype={mp_policy.reduce_dtype} | "
        f"moe={profile.is_moe} | "
        f"ac={fsdp_cfg.activation_checkpointing}"
    )
    return model


def _build_mp_policy(fsdp_cfg, profile: ModelProfile, MixedPrecisionPolicy):
    """
    Build MixedPrecisionPolicy. Supports "auto" which reads from profile.
    """
    # Resolve param_dtype
    if fsdp_cfg.param_dtype == "auto":
        param_dtype = profile.model_dtype
    else:
        param_dtype = _DTYPE_MAP.get(fsdp_cfg.param_dtype, torch.bfloat16)

    # Resolve reduce_dtype — "auto" defaults to float32 for safer gradient reduction
    # (especially important for MoE where gradients are sparse)
    if fsdp_cfg.reduce_dtype == "auto":
        reduce_dtype = torch.float32 if profile.is_moe else param_dtype
    else:
        reduce_dtype = _DTYPE_MAP.get(fsdp_cfg.reduce_dtype, torch.bfloat16)

    # Resolve output / buffer dtype
    if fsdp_cfg.buffer_dtype == "auto":
        output_dtype = param_dtype
    else:
        output_dtype = _DTYPE_MAP.get(fsdp_cfg.buffer_dtype, torch.bfloat16)

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )


def _resolve_decoder_cls(
    model: nn.Module,
    fsdp_cfg,
    profile: ModelProfile,
) -> Optional[type]:
    """
    Resolve decoder layer class.
    If fsdp_cfg.transformer_layer_cls is non-empty, use string matching
    (backward compat). Otherwise, use auto-detected class from profile.
    """
    if fsdp_cfg.transformer_layer_cls:
        # Legacy path: manual string-based matching
        found = _find_layer_classes_by_name(model, fsdp_cfg.transformer_layer_cls)
        if found:
            return found[0]
        logger.warning(
            f"Manual transformer_layer_cls {fsdp_cfg.transformer_layer_cls} "
            f"not found in model — falling back to auto-detection"
        )

    return profile.decoder_layer_cls


def _find_layer_classes_by_name(model: nn.Module, names: list[str]) -> list[type]:
    """Legacy: find layer classes by string name matching. Used only when
    the user explicitly provides transformer_layer_cls in config."""
    found = {}
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name in names and cls_name not in found:
            found[cls_name] = module.__class__
    if missing := set(names) - set(found):
        logger.warning(f"FSDP2: layer classes not found in model: {missing}")
    return list(found.values())


# ---------------------------------------------------------------------------
# Activation checkpointing — FSDP2-native API
# ---------------------------------------------------------------------------

def _apply_activation_checkpointing(
    model: nn.Module,
    profile: ModelProfile,
    decoder_cls: Optional[type],
) -> None:
    """
    Apply activation checkpointing using the FSDP2-composable API.

    Strategy:
      - Dense models: checkpoint each decoder layer
      - MoE models: checkpoint each expert module individually
        (more efficient — avoids recomputing inactive experts)

    Falls back to the legacy FSDP1 API if the composable one isn't available.
    """
    # Try FSDP2-native composable checkpoint first
    try:
        from torch.distributed._composable import checkpoint as composable_checkpoint
        _apply_ac_composable(model, profile, decoder_cls, composable_checkpoint)
        return
    except ImportError:
        pass

    # Fallback: legacy API (still works with FSDP2, just less clean)
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        _apply_ac_legacy(model, profile, decoder_cls,
                         apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
        return
    except ImportError:
        logger.warning("No activation checkpointing API available — skipping")


def _apply_ac_composable(model, profile, decoder_cls, composable_checkpoint):
    """FSDP2-native: call checkpoint() on each target module."""
    targets_applied = 0

    if profile.is_moe and profile.expert_modules:
        # MoE: checkpoint each expert individually
        for expert in profile.expert_modules:
            composable_checkpoint(expert)
            targets_applied += 1
        logger.info(
            f"AC (composable): applied to {targets_applied} expert modules"
        )
    elif decoder_cls is not None:
        # Dense: checkpoint each decoder layer
        for module in model.modules():
            if isinstance(module, decoder_cls):
                composable_checkpoint(module)
                targets_applied += 1
        logger.info(
            f"AC (composable): applied to {targets_applied} "
            f"{decoder_cls.__name__} layers"
        )
    else:
        logger.warning("AC: no target modules found — skipping")


def _apply_ac_legacy(model, profile, decoder_cls,
                     apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl):
    """Fallback: FSDP1-era API (still compatible with FSDP2)."""
    from functools import partial

    wrapper_fn = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    if profile.is_moe and profile.expert_module_cls:
        check_fn = lambda m: isinstance(m, profile.expert_module_cls)
        label = profile.expert_module_cls.__name__
    elif decoder_cls is not None:
        check_fn = lambda m: isinstance(m, decoder_cls)
        label = decoder_cls.__name__
    else:
        logger.warning("AC (legacy): no target classes — skipping")
        return

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=check_fn,
    )
    logger.info(f"AC (legacy): applied to {label}")


# ---------------------------------------------------------------------------
# DDP — MoE-aware
# ---------------------------------------------------------------------------

def _wrap_ddp(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    ddp_cfg = cfg.training.distributed.ddp

    # MoE CRITICAL: not all expert params are used every forward pass.
    # find_unused_parameters MUST be True or DDP will error.
    find_unused = ddp_cfg.find_unused_parameters
    if profile.is_moe and not find_unused:
        logger.warning(
            "MoE detected: forcing find_unused_parameters=True for DDP "
            "(not all expert params are used every forward pass)"
        )
        find_unused = True

    model = DDP(
        model.to(local_rank),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=ddp_cfg.gradient_as_bucket_view,
        broadcast_buffers=ddp_cfg.broadcast_buffers,
    )
    logger.info(f"DDP wrap complete | find_unused={find_unused}")
    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers — FSDP2 compatible
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """
    Save checkpoint. Works for FSDP2 and DDP.

    FSDP2: model.state_dict() triggers a collective all-gather.
    ALL ranks must call this simultaneously.
    Only rank 0 writes to disk.
    """
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()

    if dist.get_rank() == 0:
        torch.save(
            {"model": model_state, "optimizer": optim_state, "step": step},
            path,
        )
        logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device,
) -> int:
    """Load checkpoint. Returns the step."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    logger.info(f"Checkpoint loaded ← {path} (step={ckpt['step']})")
    return ckpt["step"]

"""
bhaskera.distributed.checkpoint
================================
Sharded checkpointing via torch.distributed.checkpoint (DCP).

Why this module replaces the old torch.save(state_dict()) flow:
    * FSDP2 `model.state_dict()` returns DTensors. torch.save on rank 0
      serialises only rank 0's local shards — other ranks' weights are lost.
      Loading such a file produces silently-corrupt weights on resume.
    * DCP writes one file per rank (sharded) and coordinates via metadata.
      Load round-trips via `set_model_state_dict` / `set_optimizer_state_dict`
      so each rank gets the right shard regardless of world size.
    * Works unchanged for DDP and single-GPU — DCP gracefully handles
      non-sharded state dicts.

On-disk layout for one checkpoint:
    <path>/
        model/...            # DCP shard files for model state
        optim/...            # DCP shard files for optimizer state
        meta.json            # {"step": int, "avg_loss": float}
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    extra: dict | None = None,
) -> None:
    """
    Save a sharded checkpoint.

    All ranks participate (DCP is collective). `path` is treated as a
    directory — it will be created if it does not exist.

    `extra` is an optional JSON-serialisable metadata dict that will be
    written only by rank 0.
    """
    path = str(path)
    Path(path).mkdir(parents=True, exist_ok=True)

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    model_sd = get_model_state_dict(model, options=options)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=options)

    dcp.save({"model": model_sd}, checkpoint_id=os.path.join(path, "model"))
    dcp.save({"optim": optim_sd}, checkpoint_id=os.path.join(path, "optim"))

    if _is_rank_zero():
        meta = {"step": int(step)}
        if extra:
            meta.update(extra)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        logger.info(f"Checkpoint saved → {path} (step={step})")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> int:
    """Load a sharded checkpoint in-place. Returns the saved step number."""
    path = str(path)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    # We must pre-populate with the current (uninitialised) state dicts so DCP
    # knows the shape and sharding of each tensor before loading into them.
    model_sd = get_model_state_dict(model, options=options)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=options)

    dcp.load({"model": model_sd}, checkpoint_id=os.path.join(path, "model"))
    dcp.load({"optim": optim_sd}, checkpoint_id=os.path.join(path, "optim"))

    set_model_state_dict(model, model_state_dict=model_sd, options=options)
    set_optimizer_state_dict(
        model,
        optimizers=optimizer,
        optim_state_dict=optim_sd,
        options=options,
    )

    step = 0
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        step = int(meta.get("step", 0))

    logger.info(f"Checkpoint loaded ← {path} (step={step})")
    return step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rank_zero() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
"""
bhaskera.distributed.wrap
=========================
Dispatcher that chooses FSDP2 or DDP based on config.
"""
from __future__ import annotations

import logging

import torch.distributed as dist
import torch.nn as nn

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


def wrap_model(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    """Wrap a model for distributed training. Strategy: 'fsdp' or 'ddp'."""
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before wrap_model(). "
            "Ray Train's TorchTrainer does this automatically; for raw SLURM "
            "make sure torch.distributed.init_process_group() has been called."
        )

    strategy = cfg.training.distributed.strategy.lower()
    if strategy == "fsdp":
        from .fsdp import wrap_fsdp2
        return wrap_fsdp2(model, cfg, local_rank, profile)
    if strategy == "ddp":
        from .ddp import wrap_ddp
        return wrap_ddp(model, cfg, local_rank, profile)
    raise ValueError(
        f"Unknown distributed strategy: '{strategy}'. Choose 'fsdp' or 'ddp'."
    )
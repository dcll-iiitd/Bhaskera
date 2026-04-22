"""
bhaskera.distributed.ddp
========================
DDP wrap. MoE-aware: forces find_unused_parameters=True for MoE so routing
to a subset of experts per forward pass doesn't crash DDP.
"""
from __future__ import annotations

import logging

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


def wrap_ddp(model: nn.Module, cfg, local_rank: int, profile: ModelProfile) -> nn.Module:
    ddp_cfg = cfg.training.distributed.ddp

    find_unused = ddp_cfg.find_unused_parameters
    if profile.is_moe and not find_unused:
        logger.warning(
            "MoE detected: forcing find_unused_parameters=True for DDP "
            "(not all expert params are used every forward pass)."
        )
        find_unused = True

    wrapped = DDP(
        model.to(local_rank),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=ddp_cfg.gradient_as_bucket_view,
        broadcast_buffers=ddp_cfg.broadcast_buffers,
    )
    logger.info(f"DDP wrap complete | find_unused={find_unused}")
    return wrapped
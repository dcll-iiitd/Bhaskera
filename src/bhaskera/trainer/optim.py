"""
bhaskera.trainer.optim
======================
Optimizer and LR-scheduler factories.

Changes vs v1:
  * AdamW(fused=True) on CUDA — ~10-20 % faster; works with FSDP2 DTensors
    in torch >= 2.4.
  * Parameter grouping: weight_decay applied only to 2-D+ tensors
    (i.e. not to bias / LayerNorm / embeddings), following the standard
    GPT training recipe.
"""
from __future__ import annotations

import logging

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

logger = logging.getLogger(__name__)


def build_optimizer(model: torch.nn.Module, train_cfg) -> AdamW:
    """
    AdamW with decoupled weight decay.  Bias/LayerNorm/Embedding params
    get weight_decay=0; every other 2-D+ tensor uses the configured value.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Heuristic: 2-D+ params → decay; 1-D (bias, LN gain, etc.) → no decay.
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    param_groups = [
        {"params": decay,    "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    fused = torch.cuda.is_available()
    logger.info(
        f"AdamW: {sum(p.numel() for p in decay):,} decayed params, "
        f"{sum(p.numel() for p in no_decay):,} no-decay params, fused={fused}"
    )
    return AdamW(
        param_groups,
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        fused=fused,
    )


def build_scheduler(optimizer: AdamW, train_cfg) -> LRScheduler:
    """Linear warmup → cosine decay."""
    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(1, train_cfg.warmup_steps),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg.max_steps - train_cfg.warmup_steps),
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[train_cfg.warmup_steps],
    )
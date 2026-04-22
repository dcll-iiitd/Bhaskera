"""
bhaskera.trainer
================
Pure training loop.  No distributed init, no Ray, no SLURM logic — all of
that lives upstream in launcher/.

Public API:
    train(model, dataset, cfg, profile, rank, local_rank, tracker) -> None
"""
from __future__ import annotations

from .loop import train

__all__ = ["train"]
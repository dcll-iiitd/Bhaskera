"""Weights & Biases logger with optional periodic GPU telemetry."""
from __future__ import annotations

import logging
from typing import Any

from .base import BaseLogger

logger = logging.getLogger(__name__)


class WandbLogger(BaseLogger):
    def __init__(self, cfg):
        import wandb
        wandb.init(
            project=cfg.logging.project,
            name=cfg.logging.run_name,
            config=cfg.as_dict(),
        )
        self._wandb = wandb
        self._every_n = max(1, cfg.logging.log_gpu_every_n_steps)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if step % self._every_n == 0:
            from bhaskera.utils.gpu_stats import gpu_stats
            metrics = {**metrics, **gpu_stats()}
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        try:
            self._wandb.finish()
        except Exception as e:
            logger.debug(f"wandb.finish failed: {e}")
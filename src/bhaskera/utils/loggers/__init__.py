"""
bhaskera.utils.loggers
======================
Experiment trackers. Factory chooses backend from config.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseLogger


def build_logger(cfg) -> Optional[BaseLogger]:
    tracker = cfg.logging.tracker
    if tracker == "wandb":
        from .wandb_logger import WandbLogger
        return WandbLogger(cfg)
    if tracker == "mlflow":
        from .mlflow_logger import MLflowLogger
        return MLflowLogger(cfg)
    if tracker in (None, "", "none"):
        return None
    raise ValueError(f"Unknown tracker '{tracker}'. Choose wandb | mlflow | null.")


__all__ = ["BaseLogger", "build_logger"]
"""MLflow logger with periodic GPU telemetry."""
from __future__ import annotations

import logging
from typing import Any

from .base import BaseLogger

logger = logging.getLogger(__name__)


class MLflowLogger(BaseLogger):
    def __init__(self, cfg):
        import mlflow
        if cfg.logging.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.logging.project)
        mlflow.start_run(run_name=cfg.logging.run_name)
        for k, v in cfg.as_dict().items():
            try:
                mlflow.log_param(k, str(v)[:500])
            except Exception:
                pass
        self._mlflow = mlflow
        self._every_n = max(1, cfg.logging.log_gpu_every_n_steps)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if step % self._every_n == 0:
            from bhaskera.utils.gpu_stats import gpu_stats
            metrics = {**metrics, **gpu_stats()}

        safe: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                safe[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not safe:
            return
        try:
            self._mlflow.log_metrics(safe, step=step)
        except Exception as e:
            logger.debug(f"mlflow.log_metrics failed: {e}")

    def finish(self) -> None:
        try:
            self._mlflow.end_run()
        except Exception as e:
            logger.debug(f"mlflow.end_run failed: {e}")
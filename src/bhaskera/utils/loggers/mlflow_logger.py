"""MLflow logger with periodic system telemetry."""
from __future__ import annotations

import logging
from typing import Any

from .base import BaseLogger

logger = logging.getLogger(__name__)


class MLflowLogger(BaseLogger):
    def __init__(self, cfg, *, rank: int = 0, world_size: int = 1) -> None:
        import mlflow
        if cfg.logging.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.logging.project)
        mlflow.start_run(run_name=cfg.logging.run_name)

        # Tag the run so multi-run sweeps can be filtered
        try:
            mlflow.set_tags({
                "rank": str(rank),
                "world_size": str(world_size),
                "framework": "bhaskera",
            })
            for tag in (getattr(cfg.logging, "tags", []) or []):
                if isinstance(tag, str) and ":" in tag:
                    k, v = tag.split(":", 1)
                    mlflow.set_tag(k.strip(), v.strip())
        except Exception:
            pass

        for k, v in cfg.as_dict().items():
            try:
                mlflow.log_param(k, str(v)[:500])
            except Exception:
                pass
        self._mlflow = mlflow
        self._every_n = max(1, cfg.logging.log_gpu_every_n_steps)
        self._rank = int(rank)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if step % self._every_n == 0:
            from bhaskera.utils.system_stats import system_stats, cuda_memory_stats
            metrics = {**metrics, **system_stats(), **cuda_memory_stats()}

        safe: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                # MLflow disallows '/' in metric names; mangle to match
                # what users see elsewhere (e.g. W&B uses '/' freely,
                # MLflow needs '_').  We keep slashes elsewhere.
                clean_k = k.replace("/", "_")
                safe[clean_k] = float(v)
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

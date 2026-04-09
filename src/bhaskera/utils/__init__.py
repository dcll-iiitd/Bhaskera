"""
bhaskera.utils
==============
Experiment loggers.

Adding a new tracker: subclass BaseLogger, implement log() and finish().
Register it in build_logger().
"""
from __future__ import annotations
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseLogger:
    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def finish(self) -> None: ...


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------

class WandbLogger(BaseLogger):
    def __init__(self, cfg):
        import wandb
        wandb.init(project=cfg.logging.project, name=cfg.logging.run_name, config=cfg.as_dict())
        self._wandb = wandb

    def log(self, metrics, step):
        self._wandb.log(metrics, step=step)

    def finish(self):
        self._wandb.finish()


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

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
        self._mlflow   = mlflow
        self._every_n  = cfg.logging.log_gpu_every_n_steps
        self._step_cnt = 0

    def log(self, metrics, step):
        self._step_cnt += 1
        if self._step_cnt % self._every_n == 0:
            metrics = {**metrics, **_gpu_stats()}
        safe = {}
        for k, v in metrics.items():
            try:
                safe[k] = float(v)
            except (TypeError, ValueError):
                pass
        if safe:
            try:
                self._mlflow.log_metrics(safe, step=step)
            except Exception:
                pass

    def finish(self):
        try:
            self._mlflow.end_run()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_logger(cfg) -> Optional[BaseLogger]:
    tracker = cfg.logging.tracker
    if tracker == "wandb":
        return WandbLogger(cfg)
    if tracker == "mlflow":
        return MLflowLogger(cfg)
    return None


# ---------------------------------------------------------------------------
# GPU stats helper
# ---------------------------------------------------------------------------

def _gpu_stats() -> dict[str, float]:
    import subprocess, xml.etree.ElementTree as ET
    try:
        out = subprocess.check_output(["nvidia-smi", "-q", "-x"], stderr=subprocess.DEVNULL, timeout=10)
        root = ET.fromstring(out)
    except Exception:
        return {}

    result = {}
    for i, gpu in enumerate(root.findall("gpu")):
        def _f(path):
            el = gpu.find(path)
            if el is None or not el.text:
                return None
            try:
                return float(el.text.strip().split()[0])
            except ValueError:
                return None

        if (v := _f("utilization/gpu_util"))   is not None: result[f"gpu/{i}/util_pct"]  = v
        if (v := _f("fb_memory_usage/used"))    is not None: result[f"gpu/{i}/mem_mib"]   = v
        if (v := _f("temperature/gpu_temp"))    is not None: result[f"gpu/{i}/temp_c"]    = v
        if (v := _f("power_readings/power_draw")) is not None: result[f"gpu/{i}/power_w"] = v
    return result

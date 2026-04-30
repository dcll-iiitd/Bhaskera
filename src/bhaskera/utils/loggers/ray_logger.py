"""
bhaskera.utils.loggers.ray_logger
=================================
Pushes training metrics into Ray's built-in Prometheus exporter so they
show up in Ray Dashboard and any Grafana that's wired to that
Prometheus.

Why this exists:
    * W&B and MLflow are great for offline experiment tracking, but
      they don't show you live cluster-wide telemetry alongside Ray's
      task/actor/object-store views.  Ray Dashboard does.
    * Every Ray worker process can publish via ``ray.util.metrics``
      with no additional infra — the dashboard agent on each node
      handles aggregation and Prometheus scrape.
    * Therefore we run this logger on **every rank** (not just rank 0)
      so per-GPU stats land in Grafana with a ``rank`` / ``gpu_id``
      tag for easy filtering.

Implementation notes:
    * Prometheus metric names must match ``[a-zA-Z_][a-zA-Z0-9_]*``.
      Bhaskera's keys are slash-separated (``gpu/0/util_pct``).  We
      translate that on the fly to ``bhaskera_gpu_util_pct`` plus a
      ``gpu_id="0"`` tag — this is the convention every Grafana JSON
      we ship expects.
    * Gauges are created lazily on first observation and cached.  The
      tag-key set is fixed at creation time, which is a Ray API
      requirement — that's why we compute it from the *first* metric
      we see.  Subsequent calls reuse the same Gauge.
    * If ``ray`` is not importable or ``ray.util.metrics`` is
      unavailable (e.g. in a unit test), the logger silently no-ops.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .base import BaseLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name / tag extraction
# ---------------------------------------------------------------------------

# Per-GPU keys: gpu/<idx>/<rest>            -> name=gpu_<rest>, tag gpu_id=<idx>
# Per-rank?  no — rank is a default tag on the logger, not in the key.
_PROM_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")
_GPU_RE = re.compile(r"^gpu/(\d+)/(.+)$")


def _parse_key(raw: str) -> tuple[str, dict[str, str]]:
    """
    Translate a slash-separated bhaskera metric key into a Prometheus
    metric name + optional dynamic tags.

    Examples:
        loss                    -> ("bhaskera_loss",                  {})
        gpu/0/util_pct          -> ("bhaskera_gpu_util_pct",          {"gpu_id": "0"})
        cuda/allocated_mib      -> ("bhaskera_cuda_allocated_mib",    {})
        sys/cpu_pct             -> ("bhaskera_sys_cpu_pct",           {})
    """
    m = _GPU_RE.match(raw)
    if m:
        gpu_id, rest = m.group(1), m.group(2)
        name = "bhaskera_gpu_" + rest
        tags = {"gpu_id": gpu_id}
    else:
        # Replace remaining slashes with underscores for the metric name
        name = "bhaskera_" + raw.replace("/", "_")
        tags = {}
    # Final sanitisation
    name = _PROM_NAME_RE.sub("_", name)
    name = name.strip("_") or "bhaskera_unnamed"
    return name, tags


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

# Module-level cache so repeat `build_logger` calls in the same process
# reuse the same Gauge objects (Ray complains otherwise).
_GAUGE_CACHE: dict[str, Any] = {}
_DEFAULT_TAG_KEYS: tuple[str, ...] = ("run_name", "project", "rank", "gpu_id")


class RayMetricsLogger(BaseLogger):
    """
    Bhaskera logger that publishes to Ray's Prometheus endpoint.

    Construct on every rank.  Pass ``rank`` / ``world_size`` so each
    sample is tagged correctly — Grafana panels can then aggregate
    or break out per-rank.
    """

    def __init__(
        self,
        cfg,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._available = self._probe_ray_metrics()
        self._rank       = int(rank)
        self._world_size = int(world_size)
        self._project    = str(getattr(cfg.logging, "project", "bhaskera"))
        self._run_name   = str(getattr(cfg.logging, "run_name", "run"))

        self._default_tags = {
            "run_name": self._run_name,
            "project":  self._project,
            "rank":     str(self._rank),
            # gpu_id will be set per-metric where applicable; Ray
            # requires the key to exist at creation time.
            "gpu_id":   "host",
        }

        if not self._available:
            logger.info(
                "RayMetricsLogger: ray.util.metrics not importable — "
                "logger is a no-op. Install ray[default] to get the "
                "metrics agent."
            )

    # ------------------------------------------------------------------
    # BaseLogger
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self._available or not metrics:
            return
        for k, v in metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            try:
                self._observe(k, fv, step)
            except Exception as e:
                # Per-metric failure should never tank the run.
                logger.debug(f"RayMetricsLogger.log({k}={v}) failed: {e}")

    def finish(self) -> None:
        # Ray gauges live as long as the worker process — nothing to flush.
        return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observe(self, key: str, value: float, step: int) -> None:
        name, dynamic_tags = _parse_key(key)
        gauge = _GAUGE_CACHE.get(name)
        if gauge is None:
            from ray.util.metrics import Gauge
            gauge = Gauge(
                name,
                description=f"Bhaskera training metric: {key}",
                tag_keys=_DEFAULT_TAG_KEYS,
            )
            _GAUGE_CACHE[name] = gauge

        tags = dict(self._default_tags)
        tags.update(dynamic_tags)
        gauge.set(value, tags=tags)

        # We also expose the global step as its own gauge so panels
        # can plot loss-vs-step (Grafana's x-axis is wall time, but a
        # `bhaskera_step` series lets users overlay).
        if key == "loss":
            self._set_step(step)

    def _set_step(self, step: int) -> None:
        gauge = _GAUGE_CACHE.get("bhaskera_step")
        if gauge is None:
            from ray.util.metrics import Gauge
            gauge = Gauge(
                "bhaskera_step",
                description="Global training step",
                tag_keys=_DEFAULT_TAG_KEYS,
            )
            _GAUGE_CACHE["bhaskera_step"] = gauge
        gauge.set(float(step), tags=self._default_tags)

    @staticmethod
    def _probe_ray_metrics() -> bool:
        try:
            import ray.util.metrics  # noqa: F401
            return True
        except Exception as e:
            logger.debug(f"ray.util.metrics import failed: {e}")
            return False

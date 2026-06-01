"""
bhaskera.serve.app
==================
Application factory: build a Ray Serve ``Application`` handle from a
``Config`` object.  This is the single call-site that translates config
into Ray Serve resource and replica settings.

Usage (inside a Ray-initialised process)::

    from ray import serve
    from bhaskera.serve.app import build_app

    app = build_app(cfg)
    serve.run(app, host=cfg.serve.host, port=cfg.serve.port)

Design notes
------------
- ``num_replicas`` and ``ray_actor_options`` come from ``cfg.serve``.
- For the HF backend ``max_ongoing_requests`` is capped at
  ``cfg.serve.hf.max_concurrent_queries`` (default 1) because
  ``model.generate()`` is not thread-safe.  Scale with replicas, not
  concurrency.  (Ray Serve ≥2.10 renamed ``max_concurrent_queries``
  to ``max_ongoing_requests``.)
- For the vLLM backend no concurrency cap is applied — ``AsyncLLMEngine``
  is designed for high concurrency and implements its own continuous batching.
- Autoscaling is activated when both ``autoscaling_min_replicas`` and
  ``autoscaling_max_replicas`` are set; otherwise a fixed replica count
  (``num_replicas``) is used.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bhaskera.config import Config

logger = logging.getLogger(__name__)


def build_app(cfg: "Config"):
    """
    Construct and return a Ray Serve application bound to ``cfg``.

    The returned object is a ``ray.serve.Application`` (a bound deployment
    handle) that can be passed directly to ``serve.run()``.

    Parameters
    ----------
    cfg:
        A fully populated :class:`bhaskera.config.Config` object.
        ``cfg.serve`` drives backend selection, replica count, and
        resource allocation.

    Returns
    -------
    ray.serve.Application
        Bound deployment handle ready for ``serve.run()``.
    """
    from .deployment import LLMDeployment

    backend  = cfg.serve.backend.lower()
    is_hf    = backend == "hf"
    is_vllm  = backend == "vllm"

    # ── Ray actor resource spec ─────────────────────────────────────────
    ray_actor_options = dict(cfg.serve.ray_actor_options)

    # Validate: vLLM is GPU-only; warn if num_gpus was accidentally set to 0.
    if is_vllm and ray_actor_options.get("num_gpus", 0) < 1:
        logger.warning(
            "serve.ray_actor_options.num_gpus=%s for vLLM backend; "
            "vLLM requires at least 1 GPU per replica.  "
            "Forcing num_gpus=1.",
            ray_actor_options.get("num_gpus"),
        )
        ray_actor_options["num_gpus"] = 1

    # ── Concurrency cap (HF only) ────────────────────────────────────────
    # Ray Serve queues excess requests rather than dispatching them
    # concurrently to the same replica, preventing race conditions inside
    # model.generate().  vLLM manages its own concurrency internally, so
    # we omit the kwarg entirely for that backend — Ray Serve ≥2.10 rejects
    # max_ongoing_requests=None with a ValueError.
    _extra: dict = {}
    if is_hf:
        _extra["max_ongoing_requests"] = cfg.serve.hf.max_concurrent_queries

    # ── Replica / autoscaling config ────────────────────────────────────
    autoscale = (
        cfg.serve.autoscaling_min_replicas is not None
        and cfg.serve.autoscaling_max_replicas is not None
    )

    if autoscale:
        from ray.serve.config import AutoscalingConfig

        autoscaling_config = AutoscalingConfig(
            min_replicas=cfg.serve.autoscaling_min_replicas,
            max_replicas=cfg.serve.autoscaling_max_replicas,
            target_num_ongoing_requests_per_replica=1 if is_hf else 10,
        )
        deployment = LLMDeployment.options(
            autoscaling_config=autoscaling_config,
            ray_actor_options=ray_actor_options,
            **_extra,
        )
        logger.info(
            "build_app | backend=%s autoscale min=%d max=%d",
            backend,
            cfg.serve.autoscaling_min_replicas,
            cfg.serve.autoscaling_max_replicas,
        )
    else:
        deployment = LLMDeployment.options(
            num_replicas=cfg.serve.num_replicas,
            ray_actor_options=ray_actor_options,
            **_extra,
        )
        logger.info(
            "build_app | backend=%s replicas=%d actor_opts=%s",
            backend, cfg.serve.num_replicas, ray_actor_options,
        )

    # Bind cfg so the deployment __init__ receives it as an argument.
    return deployment.bind(cfg)

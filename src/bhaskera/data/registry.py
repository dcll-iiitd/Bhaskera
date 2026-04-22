"""
bhaskera.data.registry
======================
Decorator-based registry for Ray-Data dataset builders.

Adding a new dataset:

    from bhaskera.data.registry import register

    @register("my_dataset")
    def _build(cfg):
        ...
        return ray.data.Dataset
"""
from __future__ import annotations

import logging
from typing import Callable

import ray.data

logger = logging.getLogger(__name__)

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a dataset builder under `name`."""
    def _wrap(fn: Callable):
        if name in REGISTRY:
            logger.warning(f"Overwriting dataset registration for '{name}'")
        REGISTRY[name] = fn
        return fn
    return _wrap


def build_ray_dataset(cfg) -> ray.data.Dataset:
    name = cfg.data.name
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(REGISTRY)}. "
            "Register yours with @register('name')."
        )
    logger.info(f"Building Ray dataset: {name}")
    return REGISTRY[name](cfg)
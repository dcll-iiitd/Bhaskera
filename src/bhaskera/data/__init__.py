"""
bhaskera.data
=============
Ray Data–native dataset pipeline. Sub-modules:

    registry   — REGISTRY, @register, build_ray_dataset
    tokenize   — TokenizerActor + tokenisation pipeline (pad-safe labels)
    datasets/  — built-in dataset builders (import triggers @register)
"""
from __future__ import annotations

try:
    from .registry import REGISTRY, register, build_ray_dataset

    # Importing the datasets package triggers each @register decorator so the
    # built-in keys ("ultrachat", "openassistant", "redpajama") appear in REGISTRY.
    from . import datasets  # noqa: F401

    __all__ = ["REGISTRY", "register", "build_ray_dataset"]
except ImportError as _e:  # Ray not installed (CPU-only / laptop mode)
    if "ray" not in str(_e).lower():
        raise
    # The Ray-backed registry is unavailable, but Ray-free submodules such as
    # ``bhaskera.data.formats`` and ``bhaskera.data.tokenize.TokenizerActor``
    # (used by launcher/tokenize_cpu.py and launcher/train_cpu.py) still import.
    __all__ = []
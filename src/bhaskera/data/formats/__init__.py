"""
bhaskera.data.formats
=====================
Pluggable "format renderers" that convert one raw row into messages or a rendered string.

A renderer signature is:

    fn(row: dict, tokenizer, options: dict) -> Union[str, List[dict]]

"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Union, List

logger = logging.getLogger(__name__)

FORMAT_REGISTRY: Dict[str, Callable[[dict, Any, dict], Union[str, List[dict]]]] = {}

_BUILTINS_LOADED = False

def register_format(name: str) -> Callable:
    def _wrap(fn: Callable) -> Callable:
        if name in FORMAT_REGISTRY:
            logger.warning(f"Overwriting format renderer for '{name}'")
        FORMAT_REGISTRY[name] = fn
        return fn
    return _wrap

def _ensure_builtins_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import builtins  # noqa: F401
    _BUILTINS_LOADED = True

def render_with_format(
    name: str,
    row: dict,
    tokenizer: Any,
    options: Optional[dict] = None,
) -> Union[str, List[dict]]:
    _ensure_builtins_loaded()
    if name not in FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown format '{name}'. "
            f"Available: {sorted(FORMAT_REGISTRY)}. "
            "Register yours with @register_format('name')."
        )
    return FORMAT_REGISTRY[name](row, tokenizer, options or {})

__all__ = [
    "FORMAT_REGISTRY",
    "register_format",
    "render_with_format",
]

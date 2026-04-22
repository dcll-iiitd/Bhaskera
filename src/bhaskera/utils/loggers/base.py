"""Base class for experiment trackers."""
from __future__ import annotations

from typing import Any


class BaseLogger:
    """Minimal logger interface.  Subclasses override log() and finish()."""

    def log(self, metrics: dict[str, Any], step: int) -> None:  # pragma: no cover
        ...

    def finish(self) -> None:  # pragma: no cover
        ...
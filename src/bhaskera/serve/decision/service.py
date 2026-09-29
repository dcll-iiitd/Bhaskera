"""The decision API without HTTP: Jev wire format in, typed answers out.

DecisionDeployment is a thin shell over these functions, so everything here is testable
without Ray, FastAPI or a model.
"""

from __future__ import annotations

from pydantic import ValidationError

from . import wire
from .schema import Request
from .translate import ALIAS_PREFIX, resolve_model, to_native, to_wire


def prepare(body: wire.SystemOneRequest, served: str) -> tuple[str, Request, dict[str, list[str]]]:
    """Model name to answer as, native request, and each question's answer keys."""
    model = resolve_model(body.model, served)
    native, keys = to_native(body)
    return model, native, keys


def finish(
    body: wire.SystemOneRequest, result: dict, keys: dict[str, list[str]], model: str
) -> tuple[dict, dict[str, str]]:
    """Wire-format payload and response headers for one engine result."""
    timing = result["timing"]
    headers = {
        "Server-Timing": (
            f"inference;dur={1000 * timing.get('inference_seconds', 0):.1f}, "
            f"total;dur={1000 * timing.get('total_seconds', 0):.1f}"
        )
    }
    return to_wire(body, result, keys, model), headers


def error_detail(error: ValueError) -> list[dict]:
    """A 422 body in the shape of FastAPI's own validation errors."""
    if isinstance(error, ValidationError):
        return [
            {"loc": ["body", *item["loc"]], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
    return [{"loc": getattr(error, "loc", ["body"]), "msg": str(error), "type": "value_error"}]


def models_payload(served: str, released: str) -> dict:
    return {
        "models": [
            {
                "name": served,
                "description": "Typed decisions read from the option logits of a model on "
                "Bhaskera; no text is generated.",
                "release_date": released,
            },
            {
                "name": f"{ALIAS_PREFIX}latest",
                "description": f"Compatibility alias: answered by {served} on this server, "
                "not by TypeSafe's Jev.",
                "release_date": released,
            },
        ]
    }


def health_payload(served: str, metadata: dict) -> dict:
    return {"status": "ready", "model": served, "engine": metadata}

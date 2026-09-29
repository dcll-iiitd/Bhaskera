"""
bhaskera.serve.decision_deployment
==================================
Ray Serve deployment of the decision backend: ``POST /v1/systemone`` in the Jev wire
format, answered from a model's answer-token logits (no generation).

One replica = one llama.cpp context (``loader.load_engine``).  Several replicas can share a
GPU (``ray_actor_options.num_gpus: auto``).  The engine scores on its own worker thread, so
handlers await it off the event loop.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request as HTTPRequest, Response
from fastapi.responses import JSONResponse
from ray import serve

from .decision import service, wire
from .decision.translate import served_name

if TYPE_CHECKING:
    from bhaskera.config import Config

logger = logging.getLogger(__name__)

_decision_app = FastAPI(
    title="Bhaskera Decision API",
    description="Typed decisions read from a model's answer-token logits, in the Jev wire format.",
    version="1.0.0",
    docs_url="/docs",
)


@_decision_app.exception_handler(ValueError)
async def _unprocessable(_: HTTPRequest, error: ValueError):
    # Engine limits (context length, question types) and wire errors, as FastAPI shapes 422s.
    return JSONResponse(status_code=422, content={"detail": service.error_detail(error)})


@serve.deployment
@serve.ingress(_decision_app)
class DecisionDeployment:
    def __init__(self, cfg: "Config") -> None:
        from .decision.loader import load_engine

        self._engine = load_engine(cfg)
        self._served = served_name(self._engine.backend.metadata)
        self._released = datetime.date.today().isoformat()
        meta = self._engine.backend.metadata
        logger.info(
            "DecisionDeployment ready | model=%s device=%s branch=%s probe_delta=%s",
            self._served, meta.get("device"), meta.get("branch_strategy"),
            meta.get("probe_branch_max_delta"),
        )

    async def _decide(self, native):
        return await asyncio.to_thread(self._engine.decide, native)

    @_decision_app.post("/v1/systemone", response_model=wire.SystemOneResponse)
    async def systemone(self, body: wire.SystemOneRequest, response: Response):
        model, native, keys = service.prepare(body, self._served)
        result = await self._decide(native)
        payload, headers = service.finish(body, result, keys, model)
        response.headers.update(headers)
        return payload

    @_decision_app.get("/v1/models", response_model=wire.ModelMetadataList)
    async def models(self):
        return service.models_payload(self._served, self._released)

    @_decision_app.get("/health")
    async def health(self):
        return service.health_payload(self._served, self._engine.backend.metadata)

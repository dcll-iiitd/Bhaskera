"""
bhaskera.serve.deployment
=========================
Ray Serve deployment that wires the OpenAI-compatible FastAPI routes to
the configured backend engine.

Architecture
------------
``_fastapi_app`` is a module-level FastAPI instance.  The ``@serve.ingress``
decorator patches it so that each HTTP request is routed to the *method* on
the live ``LLMDeployment`` actor, giving the handler access to ``self``
(and therefore the engine, tokenizer, and config).

The deployment is *not* instantiated here — callers do:

    deployment = LLMDeployment.options(num_replicas=N, ...).bind(cfg)
    serve.run(deployment, ...)

That pattern lets ``serve/app.py`` configure replica count, resource
requests, and autoscaling without touching this file.

Streaming
---------
When ``request.stream=True`` the handler returns a ``StreamingResponse``
whose body is an async generator that yields SSE-formatted JSON lines:

    data: {"id": "chatcmpl-...", "choices": [...], ...}\n\n
    data: [DONE]\n\n

Non-streaming requests collect all engine chunks and return a single JSON
body with token-count usage information.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

from .engine import BaseEngine, GenerationParams, create_engine
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    ChunkChoice,
    DeltaMessage,
    ErrorDetail,
    ErrorResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)

if TYPE_CHECKING:
    from bhaskera.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app (module-level — @serve.ingress reads this at import time)
# ---------------------------------------------------------------------------

_fastapi_app = FastAPI(
    title="Bhaskera LLM API",
    description=(
        "OpenAI-compatible `/v1/chat/completions` API powered by "
        "Bhaskera + Ray Serve.  Supports vLLM (fast-path) and HF (fallback) backends."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Global exception handler — wraps unhandled errors in OpenAI error envelope
# ---------------------------------------------------------------------------

@_fastapi_app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error in %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc),
                type="internal_server_error",
            )
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

@serve.deployment
@serve.ingress(_fastapi_app)
class LLMDeployment:
    """
    Ray Serve deployment — one replica per ``num_replicas`` setting.

    Lifecycle (per replica)
    ~~~~~~~~~~~~~~~~~~~~~~~
    ``__init__`` runs once when the replica actor is spawned:
      1. ``create_engine(cfg)`` loads the model weights.
      2. ``get_tokenizer()`` is called to obtain the chat-template tokenizer.

    For the HF backend ``create_engine`` calls ``build_model``, which may
    download from HuggingFace Hub on first run.

    Request handling
    ~~~~~~~~~~~~~~~~
    Every HTTP POST to ``/v1/chat/completions`` goes through:
      1. ``_format_prompt``  — ``apply_chat_template`` → prompt string
      2. ``_build_gen_params`` — map request fields to ``GenerationParams``
      3. engine.generate / engine.generate_full — produce tokens
      4. Build and return the OpenAI-shaped JSON (or SSE stream)
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg        = cfg
        self._engine: BaseEngine = create_engine(cfg)
        self._tokenizer  = self._engine.get_tokenizer()
        self._model_name = cfg.model.name

        # --- ADD THIS FIX ---
        if self._tokenizer.chat_template is None:
            logger.info("Injecting fallback Llama-3 chat template.")
            self._tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            )
        # --------------------

        logger.info(
            "LLMDeployment ready | backend=%s model=%s",
            cfg.serve.backend, self._model_name,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _format_prompt(self, request: ChatCompletionRequest) -> str:
        """
        Convert the ``messages`` list to a single prompt string using the
        tokenizer's chat template.

        Falls back to a plain ``<role>\ncontent</role>`` format when the
        tokenizer has no configured chat template (e.g. raw base models).
        """
        # Pydantic v2: model_dump() strips None fields by default with exclude_none.
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "apply_chat_template failed (%s); using plain role/content fallback.",
                exc,
            )
            lines = [
                f"<{m['role']}>\n{m['content']}\n</{m['role']}>"
                for m in messages
            ]
            lines.append("<assistant>")
            return "\n".join(lines)

    def _build_gen_params(self, request: ChatCompletionRequest) -> GenerationParams:
        """Map ``ChatCompletionRequest`` fields to ``GenerationParams``."""
        stop = request.stop or []
        if isinstance(stop, str):
            stop = [stop]
        return GenerationParams(
            # Fall back to the inference config default when the client omits max_tokens.
            max_new_tokens=request.max_tokens or self._cfg.inference.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=stop,
            seed=request.seed,
        )

    def _count_tokens(self, text: str) -> int:
        """Best-effort token count; used only for UsageInfo in non-streaming mode."""
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            # Never fail a request over usage-count errors.
            return 0

    # ------------------------------------------------------------------ #
    # SSE streaming helper                                               #
    # ------------------------------------------------------------------ #

    async def _stream_sse(
        self,
        req_id: str,
        prompt: str,
        params: GenerationParams,
        model_name: str,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields Server-Sent Event strings.

        Protocol (matches openai-python streaming expectations):
          - First chunk:  delta carries ``role`` only.
          - Middle chunks: delta carries ``content`` increments.
          - Final chunk:  delta is empty; ``finish_reason`` = "stop".
          - Terminator:   ``data: [DONE]``
        """
        created = int(time.time())

        # --- role chunk ------------------------------------------------
        role_chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=model_name,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        # --- content chunks -------------------------------------------
        finish_reason = "stop"
        tokens_emitted = 0
        try:
            async for token in self._engine.generate(prompt, params):
                tokens_emitted += 1
                content_chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=DeltaMessage(content=token),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"

        except Exception as exc:
            logger.exception("Engine error during streaming (req_id=%s)", req_id)
            finish_reason = "stop"

        # --- terminal chunk -------------------------------------------
        terminal_chunk = ChatCompletionChunk(
            id=req_id,
            created=created,
            model=model_name,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {terminal_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------ #
    # Routes                                                             #
    # ------------------------------------------------------------------ #

    @_fastapi_app.get("/health")
    async def health(self) -> dict:
        """Liveness probe — returns 200 as soon as the replica is ready."""
        return {
            "status": "ok",
            "model":   self._model_name,
            "backend": self._cfg.serve.backend,
        }

    @_fastapi_app.get("/v1/models")
    async def list_models(self) -> ModelList:
        """List the single served model, matching the OpenAI /v1/models spec."""
        return ModelList(
            data=[ModelCard(id=self._model_name, owned_by="bhaskera")]
        )

    @_fastapi_app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """
        Main chat endpoint, OpenAI-compatible.

        Streaming mode (``request.stream=True``):
            Returns a ``StreamingResponse`` with ``media_type="text/event-stream"``.
            The body is an SSE stream of ``ChatCompletionChunk`` JSON objects,
            terminated by ``data: [DONE]``.

        Non-streaming mode:
            Collects all engine output, counts tokens, and returns a single
            ``ChatCompletionResponse`` JSON body.
        """
        req_id  = f"chatcmpl-{uuid.uuid4().hex}"
        prompt  = self._format_prompt(request)
        params  = self._build_gen_params(request)

        logger.debug(
            "chat_completions | req_id=%s stream=%s model=%s prompt_chars=%d",
            req_id, request.stream, request.model, len(prompt),
        )

        if request.stream:
            return StreamingResponse(
                self._stream_sse(req_id, prompt, params, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control":    "no-cache",
                    "Connection":       "keep-alive",
                    # Nginx: disable proxy-level buffering so chunks reach the
                    # client immediately without waiting for the buffer to fill.
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: collect full output then respond.
        try:
            full_text = await self._engine.generate_full(prompt, params)
        except Exception as exc:
            logger.exception("Engine error (req_id=%s)", req_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompt_tokens     = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(full_text)

        return ChatCompletionResponse(
            id=req_id,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageResponse(content=full_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

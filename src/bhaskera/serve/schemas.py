"""
bhaskera.serve.schemas
======================
Pydantic v2 models that mirror the OpenAI Chat Completions API spec.

All field names, types, and defaults match the official spec so that
any OpenAI-compatible client (openai-python, LangChain, etc.) works
without modification.

References:
    https://platform.openai.com/docs/api-reference/chat/create
    https://platform.openai.com/docs/api-reference/chat/streaming
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single turn in the conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Optional sender name; passed through to apply_chat_template when present.
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""
    model: str
    messages: list[ChatMessage]

    # Sampling parameters
    temperature: float         = Field(default=1.0,  ge=0.0, le=2.0)
    top_p: float               = Field(default=1.0,  ge=0.0, le=1.0)
    top_k: int                 = Field(default=50,   ge=1)
    max_tokens: Optional[int]  = Field(default=None, ge=1)

    # Response control
    stream: bool               = False
    # One string or a list of up to 4 stop sequences.
    stop: Optional[Union[str, list[str]]] = None

    # Repetition penalties (accepted but not applied by HF backend — vLLM honours them).
    presence_penalty:  float   = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float   = Field(default=0.0, ge=-2.0, le=2.0)

    # n > 1 (parallel completions) is not supported; kept for spec compliance.
    n: int                     = Field(default=1, ge=1, le=1)

    # Reproducibility
    seed: Optional[int]        = None

    # Passthrough; not inspected by Bhaskera.
    user: Optional[str]        = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

class ChatMessageResponse(BaseModel):
    """The generated assistant turn returned in choices[].message."""
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"
    # Kept for spec compliance; not populated.
    logprobs: None = None


class UsageInfo(BaseModel):
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0


class ChatCompletionResponse(BaseModel):
    id:      str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object:  Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model:   str
    choices: list[ChatCompletionChoice]
    usage:   UsageInfo

    # Passthrough; not populated.
    system_fingerprint: Optional[str] = None


# ---------------------------------------------------------------------------
# Streaming chunks  (text/event-stream)
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    """Incremental content for a streaming chunk.

    The first chunk carries only ``role``.
    Subsequent chunks carry only ``content``.
    The final chunk has both fields empty and ``finish_reason`` set.
    """
    role:    Optional[Literal["assistant"]] = None
    content: Optional[str]                 = None


class ChunkChoice(BaseModel):
    index:         int = 0
    delta:         DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs:      None = None


class ChatCompletionChunk(BaseModel):
    id:      str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object:  Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model:   str
    choices: list[ChunkChoice]


# ---------------------------------------------------------------------------
# Model listing  (GET /v1/models)
# ---------------------------------------------------------------------------

class ModelCard(BaseModel):
    id:       str
    object:   Literal["model"] = "model"
    created:  int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "bhaskera"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data:   list[ModelCard]


# ---------------------------------------------------------------------------
# Error envelope  (returned as 4xx/5xx JSON bodies)
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    message: str
    type:    str = "internal_error"
    code:    Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail

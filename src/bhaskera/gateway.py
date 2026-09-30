import logging
import os
import time

import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse.openai import AsyncOpenAI  # <-- Switched to AsyncOpenAI

logger = logging.getLogger(__name__)

app = FastAPI(title="Bhaskera Custom Gateway")
security = HTTPBearer()

VALID_KEYS = {
    "sk-bhaskera-admin": "admin",
    "sk-bhaskera-alice": "user_alice",
    "sk-bhaskera-bob": "user_bob"
}

ray_port = os.getenv("RAY_PORT", "8000")
# Using the Async client ensures the thread isn't blocked during streaming
internal_client = AsyncOpenAI(
    base_url=f"http://127.0.0.1:{ray_port}/v1",
    api_key="internal_dummy_key"
)

# Decision backend (serve.backend: decision): plain JSON forwarding, traced per question.
decision_client = httpx.AsyncClient(
    base_url=f"http://127.0.0.1:{ray_port}", timeout=httpx.Timeout(600.0)
)


def _authorize(creds: HTTPAuthorizationCredentials) -> str:
    if creds.credentials not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return VALID_KEYS[creds.credentials]


def _trace_decision(user_id: str, body: dict, status: int, payload: dict, seconds: float) -> None:
    """One Langfuse trace per /v1/systemone call, one span per question. Never raises:
    tracing must not fail or slow a decision (the SDK ships events in the background)."""
    try:
        from langfuse import get_client, propagate_attributes

        langfuse = get_client()
        questions = body.get("questions") or {}
        with langfuse.start_as_current_observation(
            name="systemone", as_type="span",
            input={"state": body.get("state"), "questions": questions},
        ) as trace:
            with propagate_attributes(user_id=user_id, tags=["decision"]):
                for qid, answer in (payload.get("answers") or {}).items():
                    with langfuse.start_as_current_observation(
                        name=f"question:{qid}", as_type="span",
                        input=questions.get(qid), output=answer,
                    ):
                        pass
            trace.update(output={"status": status, "usage": payload.get("usage"),
                                 "gateway_seconds": seconds})
    except Exception:
        logger.debug("Langfuse tracing of a decision failed", exc_info=True)


@app.post("/v1/chat/completions")
async def chat_gateway(request: Request, creds: HTTPAuthorizationCredentials = Depends(security)):
    user_id = _authorize(creds)
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if body.get("stream"):
        # Await the async stream creation
        response_stream = await internal_client.chat.completions.create(
            **body,
            user=user_id,
        )
        
        # Use an async generator to yield chunks
        async def stream_generator():
            async for chunk in response_stream:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(
            stream_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Tells all proxies (like Cloudflare) NOT to buffer
            }
        )
        
    else:
        # Await the standard response
        response = await internal_client.chat.completions.create(
            **body,
            user=user_id,
        )
        return response


@app.post("/v1/systemone")
async def systemone_gateway(request: Request, creds: HTTPAuthorizationCredentials = Depends(security)):
    user_id = _authorize(creds)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    started = time.perf_counter()
    try:
        upstream = await decision_client.post("/v1/systemone", json=body)
        payload = upstream.json()
    except (httpx.HTTPError, ValueError) as exc:
        return JSONResponse(status_code=502,
                            content={"detail": f"decision backend unavailable: {exc!r}"})
    seconds = time.perf_counter() - started
    _trace_decision(user_id, body, upstream.status_code, payload, seconds)
    headers = {k: v for k, v in upstream.headers.items() if k.lower() == "server-timing"}
    return JSONResponse(status_code=upstream.status_code, content=payload, headers=headers)

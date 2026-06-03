from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from langfuse.openai import AsyncOpenAI  # <-- Switched to AsyncOpenAI
import os

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

@app.post("/v1/chat/completions")
async def chat_gateway(request: Request, creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    user_id = VALID_KEYS[creds.credentials]
    
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

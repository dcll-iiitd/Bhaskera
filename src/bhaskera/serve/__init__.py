"""
bhaskera.serve
==============
OpenAI-compatible LLM deployment layer, isolated from all training code.

Public surface::

    from bhaskera.serve.app import build_app
    from bhaskera.serve.engine import create_engine
    from bhaskera.serve.deployment import LLMDeployment
    from bhaskera.serve.schemas import ChatCompletionRequest, ChatCompletionResponse

Install the optional deps before importing::

    pip install "bhaskera[serve]"        # FastAPI + Ray Serve (HF backend)
    pip install "bhaskera[serve,vllm]"   # + vLLM fast-path backend
"""
__all__ = [
    "build_app",
    "create_engine",
    "LLMDeployment",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
]

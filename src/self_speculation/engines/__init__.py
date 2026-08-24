"""Inference-engine protocols and built-in adapters."""

from .base import EngineCapabilityError, InferenceEngine, validate_request
from .callable import CallableEngine, ChunkMapper, StreamFactory, default_chunk_mapper
from .openai_compatible import (
    LlamaCppEngine,
    OpenAICompatibleEngine,
    OpenAIStreamError,
    SGLangEngine,
    TGIEngine,
    VLLMEngine,
)

__all__ = [
    "CallableEngine",
    "ChunkMapper",
    "EngineCapabilityError",
    "InferenceEngine",
    "LlamaCppEngine",
    "OpenAICompatibleEngine",
    "OpenAIStreamError",
    "SGLangEngine",
    "StreamFactory",
    "TGIEngine",
    "VLLMEngine",
    "default_chunk_mapper",
    "validate_request",
]

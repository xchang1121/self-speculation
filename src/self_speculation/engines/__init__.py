"""Inference-engine protocols and built-in adapters."""

from .base import EngineCapabilityError, InferenceEngine, validate_request
from .callable import CallableEngine, ChunkMapper, StreamFactory, default_chunk_mapper

__all__ = [
    "CallableEngine",
    "ChunkMapper",
    "EngineCapabilityError",
    "InferenceEngine",
    "StreamFactory",
    "default_chunk_mapper",
    "validate_request",
]

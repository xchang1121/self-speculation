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
from .transformers import (
    TransformersBoundaryCandidateGenerator,
    TransformersDraftRequestPredicate,
    TransformersEngine,
    TransformersPromptRenderer,
)
from .vllm_native import (
    NativePromptRenderer,
    OutputMode,
    SamplingParamsFactory,
    VLLMNativeEngine,
)

__all__ = [
    "CallableEngine",
    "ChunkMapper",
    "EngineCapabilityError",
    "InferenceEngine",
    "LlamaCppEngine",
    "NativePromptRenderer",
    "OpenAICompatibleEngine",
    "OpenAIStreamError",
    "OutputMode",
    "SamplingParamsFactory",
    "SGLangEngine",
    "StreamFactory",
    "TGIEngine",
    "TransformersBoundaryCandidateGenerator",
    "TransformersDraftRequestPredicate",
    "TransformersEngine",
    "TransformersPromptRenderer",
    "VLLMEngine",
    "VLLMNativeEngine",
    "default_chunk_mapper",
    "validate_request",
]

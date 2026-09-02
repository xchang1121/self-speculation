"""Inference-engine protocols and built-in adapters."""

from .base import (
    EngineCapabilityError,
    EngineContextLimitError,
    InferenceEngine,
    PromptCountingEngine,
    RequestContextBudget,
    fit_request_to_context,
    validate_request,
)
from .callable import CallableEngine, ChunkMapper, StreamFactory, default_chunk_mapper
from .openai_compatible import (
    LlamaCppEngine,
    OpenAICompatibleEngine,
    OpenAIStreamError,
    PromptTokenCounter,
    SGLangEngine,
    TGIEngine,
    VLLMEngine,
)
from .llama_cpp_native import (
    LlamaCppBoundaryDraftModel,
    LlamaCppDraftRequestPredicate,
    LlamaCppPromptRenderer,
    LlamaCppPythonEngine,
)
from .transformers import (
    TransformersBoundaryCandidateGenerator,
    TransformersDraftRequestPredicate,
    TransformersEngine,
    TransformersPromptRenderer,
)
from .vllm_native import (
    NativePromptRenderer,
    NativePromptTokenCounter,
    OutputMode,
    SamplingParamsFactory,
    VLLMNativeEngine,
)

__all__ = [
    "CallableEngine",
    "ChunkMapper",
    "EngineCapabilityError",
    "EngineContextLimitError",
    "InferenceEngine",
    "LlamaCppEngine",
    "LlamaCppBoundaryDraftModel",
    "LlamaCppDraftRequestPredicate",
    "LlamaCppPromptRenderer",
    "LlamaCppPythonEngine",
    "NativePromptRenderer",
    "NativePromptTokenCounter",
    "OpenAICompatibleEngine",
    "OpenAIStreamError",
    "OutputMode",
    "PromptCountingEngine",
    "PromptTokenCounter",
    "RequestContextBudget",
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
    "fit_request_to_context",
    "validate_request",
]

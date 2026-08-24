"""Engine-agnostic streaming self-speculation for tool calls."""

from .engines import EngineCapabilityError, InferenceEngine, validate_request
from .forks import (
    CallableForkBuilder,
    ForkBuildError,
    ForkRequestBuilder,
    PrefixForkBuilder,
)
from .models import (
    EngineCapabilities,
    InferenceRequest,
    StreamChunk,
    StreamSnapshot,
    TokenLogprob,
    ToolCall,
    ToolCallDelta,
)

__version__ = "0.1.0"

__all__ = [
    "EngineCapabilities",
    "EngineCapabilityError",
    "CallableForkBuilder",
    "ForkBuildError",
    "ForkRequestBuilder",
    "InferenceEngine",
    "InferenceRequest",
    "PrefixForkBuilder",
    "StreamChunk",
    "StreamSnapshot",
    "TokenLogprob",
    "ToolCall",
    "ToolCallDelta",
    "__version__",
    "validate_request",
]

"""Engine-agnostic streaming self-speculation for tool calls."""

from .controller import ForkController, ForkRunResult, ForkTrigger, first_output_trigger
from .decoding import DecoderFactory, ToolCallDecoder
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
from .events import (
    ForkChunkEvent,
    ForkCompletedEvent,
    ForkEvent,
    ForkFailedEvent,
    ForkSkippedEvent,
    ForkStartedEvent,
    MainChunkEvent,
    MainCompletedEvent,
    ToolCallEvent,
)

__version__ = "0.1.0"

__all__ = [
    "EngineCapabilities",
    "EngineCapabilityError",
    "DecoderFactory",
    "CallableForkBuilder",
    "ForkBuildError",
    "ForkChunkEvent",
    "ForkCompletedEvent",
    "ForkController",
    "ForkEvent",
    "ForkFailedEvent",
    "ForkRequestBuilder",
    "ForkRunResult",
    "ForkSkippedEvent",
    "ForkStartedEvent",
    "ForkTrigger",
    "InferenceEngine",
    "InferenceRequest",
    "MainChunkEvent",
    "MainCompletedEvent",
    "PrefixForkBuilder",
    "StreamChunk",
    "StreamSnapshot",
    "TokenLogprob",
    "ToolCall",
    "ToolCallDecoder",
    "ToolCallDelta",
    "ToolCallEvent",
    "__version__",
    "first_output_trigger",
    "validate_request",
]

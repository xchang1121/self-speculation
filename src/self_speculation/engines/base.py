"""The minimal protocol required to plug in an inference engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..models import EngineCapabilities, InferenceRequest, StreamChunk


class EngineCapabilityError(ValueError):
    """Raised before dispatch when an engine cannot serve a request form."""


class EngineContextLimitError(EngineCapabilityError):
    """Raised when an exact prompt leaves no room for a fork completion."""


@dataclass(frozen=True, slots=True)
class RequestContextBudget:
    """Exact prompt and output allowance for one bounded request."""

    prompt_tokens: int
    max_context_tokens: int
    max_output_tokens: int


@runtime_checkable
class InferenceEngine(Protocol):
    """Any engine with this one async-stream method can be forked."""

    name: str
    capabilities: EngineCapabilities

    def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        ...


@runtime_checkable
class PromptCountingEngine(Protocol):
    """Optional engine capability used to avoid overflow-triggered compaction."""

    async def prompt_token_count(self, request: InferenceRequest) -> int:
        ...


def validate_request(engine: InferenceEngine, request: InferenceRequest) -> None:
    if not engine.capabilities.supports(request):
        raise EngineCapabilityError(
            f"engine {engine.name!r} does not support {request.input_mode} requests"
        )


async def fit_request_to_context(
    engine: InferenceEngine,
    request: InferenceRequest,
) -> tuple[InferenceRequest, RequestContextBudget | None]:
    """Clamp only output length, preserving the request's exact input prefix.

    Engines opt in by declaring ``max_context_tokens`` and implementing exact
    prompt token counting. An unknown window remains unmodified; a declared
    window without a counter is rejected before it can trigger provider-side
    truncation or compaction.
    """

    max_context_tokens = engine.capabilities.max_context_tokens
    if max_context_tokens is None:
        return request, None
    if not isinstance(engine, PromptCountingEngine):
        raise EngineCapabilityError(
            f"engine {engine.name!r} declares a context window but cannot count "
            "the rendered prompt"
        )
    prompt_tokens = int(await engine.prompt_token_count(request))
    if prompt_tokens < 0:
        raise ValueError("prompt_token_count must be non-negative")
    remaining = max_context_tokens - prompt_tokens
    if remaining <= 0:
        raise EngineContextLimitError(
            f"engine {engine.name!r} fork prompt uses {prompt_tokens} tokens, "
            f"exceeding its {max_context_tokens}-token context window"
        )
    max_output_tokens = min(request.max_tokens or remaining, remaining)
    return (
        request.with_changes(max_tokens=max_output_tokens),
        RequestContextBudget(
            prompt_tokens=prompt_tokens,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
        ),
    )

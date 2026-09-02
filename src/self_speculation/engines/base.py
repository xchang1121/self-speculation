"""The minimal protocol required to plug in an inference engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..models import EngineCapabilities, InferenceRequest, StreamChunk


class EngineCapabilityError(ValueError):
    """Raised before dispatch when an engine cannot serve a request form."""


class EngineContextLimitError(EngineCapabilityError):
    """Raised when an exact prompt leaves no room for a fork completion."""


class EngineCacheReuseError(EngineCapabilityError):
    """Raised when a fork cannot prove that it reused Actor KV state."""


ForkCachePolicy = Literal["required", "prefer", "off"]


@dataclass(frozen=True, slots=True)
class ForkCacheEvidence:
    """Per-request proof derived from normalized engine cache accounting."""

    policy: ForkCachePolicy
    configured: bool
    reported: bool
    reused_tokens: int
    prompt_tokens: int | None = None

    @property
    def verified(self) -> bool:
        return self.reported and self.reused_tokens > 0

    def to_mapping(self) -> dict[str, int | float | bool | str | None]:
        return {
            "policy": self.policy,
            "configured": self.configured,
            "reported": self.reported,
            "verified": self.verified,
            "reused_tokens": self.reused_tokens,
            "prompt_tokens": self.prompt_tokens,
            "hit_rate": (
                self.reused_tokens / self.prompt_tokens
                if self.prompt_tokens
                else None
            ),
        }


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


def validate_fork_cache(
    engine: InferenceEngine,
    policy: ForkCachePolicy,
) -> None:
    if policy not in ("required", "prefer", "off"):
        raise ValueError("fork cache policy must be required, prefer, or off")
    if policy != "required":
        return
    if engine.capabilities.prefix_cache is not True:
        raise EngineCacheReuseError(
            f"engine {engine.name!r} does not guarantee prefix caching"
        )
    if not engine.capabilities.cache_read_reporting:
        raise EngineCacheReuseError(
            f"engine {engine.name!r} cannot report per-request cache reuse"
        )


def verify_fork_cache(
    engine: InferenceEngine,
    policy: ForkCachePolicy,
    *,
    reused_tokens: int,
    prompt_tokens: int | None,
) -> ForkCacheEvidence:
    evidence = ForkCacheEvidence(
        policy=policy,
        configured=engine.capabilities.prefix_cache is True,
        reported=engine.capabilities.cache_read_reporting,
        reused_tokens=max(0, reused_tokens),
        prompt_tokens=prompt_tokens,
    )
    if policy == "required" and not evidence.verified:
        raise EngineCacheReuseError(
            f"engine {engine.name!r} reported no KV-cache reuse for the fork"
        )
    return evidence


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

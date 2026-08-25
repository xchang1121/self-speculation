"""Portable protocol for feeding a speculative action back to an engine."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from ..models import InferenceRequest, ToolCall


DraftFormatter = Callable[[tuple[ToolCall, ...]], str | Awaitable[str]]
DraftTokenizer = Callable[[str], list[int] | tuple[int, ...] | Awaitable[list[int] | tuple[int, ...]]]
PromptLengthResolver = Callable[[InferenceRequest], int | Awaitable[int]]
BoundaryResolver = Callable[[tuple[ToolCall, ...]], "DraftBoundary | None"]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(frozen=True, slots=True)
class DraftBoundary:
    """Tool-call boundary at which an engine should offer the draft."""

    text: str | None = None
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.text and not self.token_ids:
            raise ValueError("draft boundary needs text or token_ids")


@dataclass(frozen=True, slots=True)
class DraftRequest:
    """A speculative continuation registered against one active main request."""

    request_id: str
    text: str = ""
    token_ids: tuple[int, ...] = ()
    boundary: DraftBoundary | None = None
    prompt_token_count: int | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not self.request_id.strip():
            raise ValueError("draft request_id must not be empty")
        if not self.text and not self.token_ids:
            raise ValueError("draft needs text or token_ids")
        if self.prompt_token_count is not None and self.prompt_token_count < 0:
            raise ValueError("prompt_token_count must be non-negative")


@dataclass(frozen=True, slots=True)
class DraftBundle:
    """Ordered alternative continuations for one active main request."""

    request_id: str
    drafts: tuple[DraftRequest, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "drafts", tuple(self.drafts))
        if not self.request_id.strip():
            raise ValueError("draft bundle request_id must not be empty")
        if not self.drafts:
            raise ValueError("draft bundle must contain at least one draft")
        if any(draft.request_id != self.request_id for draft in self.drafts):
            raise ValueError("every bundled draft must use the bundle request_id")


@dataclass(frozen=True, slots=True)
class DraftReceipt:
    request_id: str
    registered: bool
    draft_token_count: int | None = None
    accepted_token_count: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DraftFeedback(Protocol):
    """Optional side channel implemented by D3-capable engines."""

    name: str

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        ...

    async def clear(self, request_id: str) -> None:
        ...


@runtime_checkable
class DraftBundleFeedback(Protocol):
    """Optional multi-candidate extension to :class:`DraftFeedback`."""

    name: str

    async def submit_bundle(self, bundle: DraftBundle) -> DraftReceipt:
        ...

    async def clear(self, request_id: str) -> None:
        ...


@runtime_checkable
class DraftBuilder(Protocol):
    async def build(
        self,
        tool_calls: tuple[ToolCall, ...],
        main_request: InferenceRequest,
        fork_request: InferenceRequest,
    ) -> DraftRequest:
        ...


@dataclass(slots=True)
class ToolCallDraftBuilder:
    formatter: DraftFormatter
    tokenizer: DraftTokenizer | None = None
    boundary_resolver: BoundaryResolver | None = None
    prompt_length_resolver: PromptLengthResolver | None = None
    max_draft_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_draft_tokens is not None and self.max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")

    async def build(
        self,
        tool_calls: tuple[ToolCall, ...],
        main_request: InferenceRequest,
        fork_request: InferenceRequest,
    ) -> DraftRequest:
        prompt_token_count: int | None = None
        if self.prompt_length_resolver is not None:
            prompt_token_count = int(
                await _resolve(self.prompt_length_resolver(main_request))
            )
        elif "prompt_token_count" in main_request.extra:
            prompt_token_count = int(main_request.extra["prompt_token_count"])

        return await self.build_for_request(
            tool_calls,
            request_id=main_request.request_id,
            prompt_token_count=prompt_token_count,
            metadata={
                "fork_request_id": fork_request.request_id,
                "formats": tuple(call.format for call in tool_calls),
            },
        )

    async def build_for_request(
        self,
        tool_calls: tuple[ToolCall, ...],
        *,
        request_id: str,
        prompt_token_count: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DraftRequest:
        """Build a concrete-action draft without requiring a fork request."""

        if not tool_calls:
            raise ValueError("cannot build a draft without tool calls")
        text = await _resolve(self.formatter(tool_calls))
        if not isinstance(text, str) or not text:
            raise TypeError("draft formatter must return a non-empty string")

        token_ids: tuple[int, ...] = ()
        if self.tokenizer is not None:
            encoded = await _resolve(self.tokenizer(text))
            token_ids = tuple(int(item) for item in encoded)
            if self.max_draft_tokens is not None:
                token_ids = token_ids[: self.max_draft_tokens]

        boundary = (
            self.boundary_resolver(tool_calls)
            if self.boundary_resolver is not None
            else None
        )
        if (
            boundary is not None
            and boundary.text
            and not boundary.token_ids
            and self.tokenizer is not None
        ):
            encoded_boundary = tuple(
                int(item) for item in await _resolve(self.tokenizer(boundary.text))
            )
            if encoded_boundary:
                boundary = DraftBoundary(
                    text=boundary.text,
                    token_ids=encoded_boundary,
                )
        return DraftRequest(
            request_id=request_id,
            text=text,
            token_ids=token_ids,
            boundary=boundary,
            prompt_token_count=prompt_token_count,
            tool_calls=tool_calls,
            metadata={
                "formats": tuple(call.format for call in tool_calls),
                **dict(metadata or {}),
            },
        )

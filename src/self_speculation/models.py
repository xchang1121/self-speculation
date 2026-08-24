"""Shared, dependency-free data models used by every engine adapter."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping
from uuid import uuid4


JsonValue = Any


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """A normalized request accepted by a streaming inference engine.

    Exactly one input form is required: ``prompt`` for a raw completion or
    ``messages`` for a chat completion. Engine-specific options belong in
    ``extra`` so the core does not need to know every serving API.
    """

    prompt: str | None = None
    messages: tuple[Mapping[str, Any], ...] = ()
    model: str | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    request_id: str = field(default_factory=lambda: uuid4().hex)
    max_tokens: int | None = None
    temperature: float | None = 0.0
    stop: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "stop", tuple(self.stop))

        has_prompt = self.prompt is not None
        has_messages = bool(self.messages)
        if has_prompt == has_messages:
            raise ValueError("exactly one of prompt or messages must be provided")
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if any(not value for value in self.stop):
            raise ValueError("stop strings must not be empty")

    @property
    def input_mode(self) -> str:
        return "prompt" if self.prompt is not None else "chat"

    def with_changes(self, **changes: Any) -> "InferenceRequest":
        """Return a validated copy while preserving unspecified fields."""

        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token: str
    logprob: float | None = None
    top_logprobs: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One provider-neutral fragment of a structured tool call."""

    index: int = 0
    call_id: str | None = None
    name: str = ""
    arguments: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("tool-call delta index must be non-negative")


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One normalized streaming update.

    ``reasoning`` and ``text`` remain separate for consumers that hide chain of
    thought. ``generated_text`` joins them in generation order for prefix
    forking. Adapters should place raw completion text in ``text`` and mapped
    reasoning fields (for example ``reasoning_content``) in ``reasoning``.
    """

    text: str = ""
    reasoning: str = ""
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[TokenLogprob, ...] = ()
    finish_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    raw: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_call_deltas", tuple(self.tool_call_deltas))
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "logprobs", tuple(self.logprobs))

    @property
    def generated_text(self) -> str:
        return self.reasoning + self.text

    @property
    def has_output(self) -> bool:
        return bool(
            self.generated_text
            or self.tool_call_deltas
            or self.token_ids
            or self.finish_reason
        )


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    """Immutable accumulated state of a stream at a possible fork point."""

    generated_text: str = ""
    content: str = ""
    reasoning: str = ""
    chunk_count: int = 0
    output_chunk_count: int = 0
    token_count: int = 0
    finish_reason: str | None = None

    def append(self, chunk: StreamChunk) -> "StreamSnapshot":
        return StreamSnapshot(
            generated_text=self.generated_text + chunk.generated_text,
            content=self.content + chunk.text,
            reasoning=self.reasoning + chunk.reasoning,
            chunk_count=self.chunk_count + 1,
            output_chunk_count=self.output_chunk_count + int(chunk.has_output),
            token_count=self.token_count + len(chunk.token_ids),
            finish_reason=chunk.finish_reason or self.finish_reason,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A completed tool action decoded from text or structured deltas."""

    name: str
    arguments: JsonValue = field(default_factory=dict)
    call_id: str | None = None
    index: int = 0
    format: str = "unknown"
    raw: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool-call name must not be empty")
        if self.index < 0:
            raise ValueError("tool-call index must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Feature declaration used for early compatibility checks."""

    prompt: bool = True
    chat: bool = False
    structured_tool_deltas: bool = False
    token_ids: bool = False
    logprobs: bool = False
    prefix_cache: bool | None = None
    draft_feedback: bool = False

    def supports(self, request: InferenceRequest) -> bool:
        return self.prompt if request.input_mode == "prompt" else self.chat

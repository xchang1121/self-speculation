"""Typed events emitted while a main stream and its fork run concurrently."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .drafts import DraftReceipt, DraftRequest
from .models import InferenceRequest, StreamChunk, StreamSnapshot, ToolCall


class ForkEvent:
    kind: ClassVar[str]


@dataclass(frozen=True, slots=True)
class MainChunkEvent(ForkEvent):
    kind: ClassVar[str] = "main_chunk"
    chunk: StreamChunk
    snapshot: StreamSnapshot


@dataclass(frozen=True, slots=True)
class ForkStartedEvent(ForkEvent):
    kind: ClassVar[str] = "fork_started"
    request: InferenceRequest
    snapshot: StreamSnapshot


@dataclass(frozen=True, slots=True)
class ForkChunkEvent(ForkEvent):
    kind: ClassVar[str] = "fork_chunk"
    chunk: StreamChunk


@dataclass(frozen=True, slots=True)
class ToolCallEvent(ForkEvent):
    kind: ClassVar[str] = "tool_call"
    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class DraftSubmittedEvent(ForkEvent):
    kind: ClassVar[str] = "draft_submitted"
    draft: DraftRequest
    receipt: DraftReceipt


@dataclass(frozen=True, slots=True)
class DraftClearedEvent(ForkEvent):
    kind: ClassVar[str] = "draft_cleared"
    request_id: str


@dataclass(frozen=True, slots=True)
class DraftFailedEvent(ForkEvent):
    kind: ClassVar[str] = "draft_failed"
    stage: str
    error: Exception


@dataclass(frozen=True, slots=True)
class MainCompletedEvent(ForkEvent):
    kind: ClassVar[str] = "main_completed"
    snapshot: StreamSnapshot


@dataclass(frozen=True, slots=True)
class ForkCompletedEvent(ForkEvent):
    kind: ClassVar[str] = "fork_completed"
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class ForkSkippedEvent(ForkEvent):
    kind: ClassVar[str] = "fork_skipped"
    reason: str


@dataclass(frozen=True, slots=True)
class ForkFailedEvent(ForkEvent):
    kind: ClassVar[str] = "fork_failed"
    stage: str
    error: Exception

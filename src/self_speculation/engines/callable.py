"""Adapter for arbitrary application or inference-engine callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..models import (
    EngineCapabilities,
    InferenceRequest,
    StreamChunk,
    TokenLogprob,
    ToolCallDelta,
)


RawStream = AsyncIterable[Any] | Iterable[Any]
StreamFactory = Callable[
    [InferenceRequest],
    RawStream | Awaitable[RawStream],
]
ChunkMapper = Callable[[Any], StreamChunk | Awaitable[StreamChunk]]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _tool_delta(value: Any) -> ToolCallDelta:
    if isinstance(value, ToolCallDelta):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("tool_call_deltas entries must be mappings or ToolCallDelta")
    return ToolCallDelta(
        index=int(value.get("index", 0)),
        call_id=value.get("call_id") or value.get("id"),
        name=str(value.get("name", "")),
        arguments=str(value.get("arguments", "")),
    )


def _logprob(value: Any) -> TokenLogprob:
    if isinstance(value, TokenLogprob):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("logprobs entries must be mappings or TokenLogprob")
    return TokenLogprob(
        token=str(value.get("token", "")),
        logprob=value.get("logprob"),
        top_logprobs=dict(value.get("top_logprobs") or {}),
    )


def default_chunk_mapper(value: Any) -> StreamChunk:
    if isinstance(value, StreamChunk):
        return value
    if isinstance(value, str):
        return StreamChunk(text=value)
    if not isinstance(value, Mapping):
        raise TypeError(
            "stream items must be StreamChunk, str, mapping, or use a custom mapper"
        )
    return StreamChunk(
        text=str(value.get("text", "")),
        reasoning=str(value.get("reasoning", "")),
        tool_call_deltas=tuple(
            _tool_delta(item) for item in value.get("tool_call_deltas", ())
        ),
        token_ids=tuple(int(item) for item in value.get("token_ids", ())),
        logprobs=tuple(_logprob(item) for item in value.get("logprobs", ())),
        finish_reason=value.get("finish_reason"),
        usage=dict(value.get("usage") or {}),
        raw=value,
    )


_END = object()


def _next_or_end(iterator: Iterable[Any]) -> Any:
    return next(iterator, _END)  # type: ignore[arg-type]


@dataclass(slots=True)
class CallableEngine:
    """Turn a sync/async stream factory into an ``InferenceEngine``.

    Synchronous iterators are advanced in worker threads so a blocking native
    engine does not stall the controller's asyncio loop.
    """

    factory: StreamFactory
    mapper: ChunkMapper = default_chunk_mapper
    name: str = "callable"
    capabilities: EngineCapabilities = field(
        default_factory=lambda: EngineCapabilities(prompt=True, chat=True)
    )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        source = await _resolve(self.factory(request))
        if isinstance(source, AsyncIterable):
            async for item in source:
                chunk = await _resolve(self.mapper(item))
                if not isinstance(chunk, StreamChunk):
                    raise TypeError("chunk mapper must return StreamChunk")
                yield chunk
            return

        if isinstance(source, Iterable):
            iterator = iter(source)
            try:
                while True:
                    item = await asyncio.to_thread(_next_or_end, iterator)
                    if item is _END:
                        break
                    chunk = await _resolve(self.mapper(item))
                    if not isinstance(chunk, StreamChunk):
                        raise TypeError("chunk mapper must return StreamChunk")
                    yield chunk
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    await asyncio.to_thread(close)
            return

        raise TypeError("stream factory must return an iterable or async iterable")

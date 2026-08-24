"""Protocol consumed by the fork controller, independent of parser format."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Protocol, runtime_checkable

from ..models import StreamChunk, ToolCall


@runtime_checkable
class ToolCallDecoder(Protocol):
    """A fresh stateful decoder is created for each speculative fork."""

    def feed(self, chunk: StreamChunk) -> Iterable[ToolCall]:
        ...

    def finish(self) -> Iterable[ToolCall]:
        ...


DecoderFactory = Callable[[], ToolCallDecoder]

"""Parser registry and a streaming decoder shared by all tool-call formats."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ..models import StreamChunk, ToolCall, ToolCallDelta


@runtime_checkable
class TextToolCallParser(Protocol):
    """Stateless parser over the complete text observed so far."""

    name: str

    def parse(self, text: str) -> Iterable[ToolCall]:
        ...


ParserFactory = Callable[[], TextToolCallParser]


class ParserRegistrationError(ValueError):
    pass


class ParserRegistry:
    """Ordered parser factories; order is the auto-detection priority."""

    def __init__(self) -> None:
        self._factories: OrderedDict[str, ParserFactory] = OrderedDict()

    def register(
        self,
        name: str,
        factory: ParserFactory,
        *,
        replace: bool = False,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ParserRegistrationError("parser name must not be empty")
        if normalized in self._factories and not replace:
            raise ParserRegistrationError(f"parser {normalized!r} is already registered")
        self._factories[normalized] = factory

    def unregister(self, name: str) -> None:
        try:
            del self._factories[name.strip().lower()]
        except KeyError as error:
            raise ParserRegistrationError(f"unknown parser {name!r}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def create_parsers(
        self,
        names: str | Sequence[str] = "auto",
    ) -> tuple[TextToolCallParser, ...]:
        if names == "auto":
            selected = self.names()
        elif isinstance(names, str):
            selected = (names,)
        else:
            selected = tuple(names)
        parsers: list[TextToolCallParser] = []
        for name in selected:
            normalized = name.strip().lower()
            try:
                parser = self._factories[normalized]()
            except KeyError as error:
                raise ParserRegistrationError(f"unknown parser {name!r}") from error
            if not isinstance(parser, TextToolCallParser):
                raise ParserRegistrationError(
                    f"factory for {normalized!r} did not return TextToolCallParser"
                )
            parsers.append(parser)
        return tuple(parsers)

    def decoder(
        self,
        names: str | Sequence[str] = "auto",
        *,
        initial_text: str = "",
        max_buffer_chars: int = 1_000_000,
    ) -> "StreamingToolCallDecoder":
        return StreamingToolCallDecoder(
            self.create_parsers(names),
            initial_text=initial_text,
            max_buffer_chars=max_buffer_chars,
        )


@dataclass(slots=True)
class _StructuredState:
    call_id: str | None = None
    name: str = ""
    arguments: str = ""


def _merge_fragment(current: str, fragment: str) -> str:
    if not fragment:
        return current
    if not current or fragment.startswith(current):
        return fragment
    if current.endswith(fragment):
        return current
    return current + fragment


def _fingerprint(tool_call: ToolCall) -> tuple[object, ...]:
    try:
        arguments = json.dumps(
            tool_call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        arguments = repr(tool_call.arguments)
    return (
        tool_call.format,
        tool_call.index,
        tool_call.call_id,
        tool_call.name,
        arguments,
    )


class StreamingToolCallDecoder:
    """Decode structured deltas or auto-detect one registered text branch."""

    def __init__(
        self,
        parsers: Sequence[TextToolCallParser] = (),
        *,
        initial_text: str = "",
        max_buffer_chars: int = 1_000_000,
    ) -> None:
        if max_buffer_chars <= 0:
            raise ValueError("max_buffer_chars must be positive")
        self.parsers = tuple(parsers)
        self.buffer = initial_text
        self.max_buffer_chars = max_buffer_chars
        self._locked_parser: TextToolCallParser | None = None
        self._structured: dict[int, _StructuredState] = {}
        self._structured_seen = False
        self._emitted: set[tuple[object, ...]] = set()
        self._finished = False

    def _new(self, calls: Iterable[ToolCall]) -> tuple[ToolCall, ...]:
        fresh: list[ToolCall] = []
        for tool_call in calls:
            key = _fingerprint(tool_call)
            if key not in self._emitted:
                self._emitted.add(key)
                fresh.append(tool_call)
        return tuple(fresh)

    def _apply_delta(self, delta: ToolCallDelta) -> None:
        state = self._structured.setdefault(delta.index, _StructuredState())
        if delta.call_id:
            state.call_id = delta.call_id
        state.name = _merge_fragment(state.name, delta.name)
        state.arguments = _merge_fragment(state.arguments, delta.arguments)

    def _structured_calls(self, *, final: bool) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index, state in sorted(self._structured.items()):
            if not state.name:
                continue
            raw_arguments = state.arguments.strip()
            if not raw_arguments:
                if not final:
                    continue
                arguments: object = {}
            else:
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    continue
                if not isinstance(arguments, (dict, list)) and not final:
                    continue
            calls.append(
                ToolCall(
                    name=state.name,
                    arguments=arguments,
                    call_id=state.call_id,
                    index=index,
                    format="structured",
                    raw=state.arguments,
                )
            )
        return self._new(calls)

    def _text_calls(self) -> tuple[ToolCall, ...]:
        if self._structured_seen:
            return ()
        candidates = (
            (self._locked_parser,)
            if self._locked_parser is not None
            else self.parsers
        )
        for parser in candidates:
            if parser is None:
                continue
            calls = tuple(parser.parse(self.buffer))
            if calls:
                self._locked_parser = parser
                return self._new(calls)
        return ()

    def feed(self, chunk: StreamChunk) -> Iterable[ToolCall]:
        if self._finished:
            raise RuntimeError("cannot feed a finished decoder")
        self.buffer += chunk.generated_text
        if len(self.buffer) > self.max_buffer_chars:
            raise ValueError("tool-call decode buffer exceeded max_buffer_chars")
        if chunk.tool_call_deltas:
            self._structured_seen = True
            for delta in chunk.tool_call_deltas:
                self._apply_delta(delta)
        structured = self._structured_calls(final=chunk.finish_reason is not None)
        return structured or self._text_calls()

    def finish(self) -> Iterable[ToolCall]:
        if self._finished:
            return ()
        self._finished = True
        structured = self._structured_calls(final=True)
        return structured or self._text_calls()

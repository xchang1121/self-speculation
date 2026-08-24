"""Async orchestration for one main stream and one speculative fork."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from .decoding import DecoderFactory, ToolCallDecoder
from .engines import InferenceEngine, validate_request
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
from .forks import ForkRequestBuilder
from .models import InferenceRequest, StreamChunk, StreamSnapshot, ToolCall


ForkTrigger = Callable[[StreamSnapshot, StreamChunk], bool]


def first_output_trigger(snapshot: StreamSnapshot, chunk: StreamChunk) -> bool:
    """Match SPORK D1: fork after the first generated token/useful delta."""

    del snapshot
    return bool(chunk.generated_text or chunk.token_ids or chunk.tool_call_deltas)


@dataclass(frozen=True, slots=True)
class ForkRunResult:
    main: StreamSnapshot
    fork_request: InferenceRequest | None
    fork_text: str
    tool_calls: tuple[ToolCall, ...]
    skipped_reason: str | None = None
    failure: ForkFailedEvent | None = None


async def _close(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


class ForkController:
    """Fork an inference stream once and decode the branch concurrently.

    Main-stream exceptions remain fatal. Fork build/stream/decode failures are
    best-effort by default: they emit ``ForkFailedEvent`` while the main stream
    continues unchanged. Set ``strict_fork_errors`` to surface them instead.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        builder: ForkRequestBuilder,
        decoder_factory: DecoderFactory,
        *,
        fork_engine: InferenceEngine | None = None,
        trigger: ForkTrigger = first_output_trigger,
        strict_fork_errors: bool = False,
    ) -> None:
        self.engine = engine
        self.fork_engine = fork_engine or engine
        self.builder = builder
        self.decoder_factory = decoder_factory
        self.trigger = trigger
        self.strict_fork_errors = strict_fork_errors

    async def stream(self, request: InferenceRequest) -> AsyncIterator[ForkEvent]:
        validate_request(self.engine, request)

        main_iterator = self.engine.stream(request).__aiter__()
        fork_iterator: AsyncIterator[StreamChunk] | None = None
        main_task: asyncio.Task[StreamChunk] | None = asyncio.create_task(
            anext(main_iterator)
        )
        fork_task: asyncio.Task[StreamChunk] | None = None
        snapshot = StreamSnapshot()
        fork_started = False
        fork_terminal = False
        decoder: ToolCallDecoder | None = None
        decoded: list[ToolCall] = []

        try:
            while main_task is not None or fork_task is not None:
                tasks = {task for task in (main_task, fork_task) if task is not None}
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Process main first when both complete in the same loop. This
                # keeps snapshots deterministic and opens the fork immediately.
                if main_task is not None and main_task in done:
                    try:
                        chunk = main_task.result()
                    except StopAsyncIteration:
                        main_task = None
                        yield MainCompletedEvent(snapshot)
                        if not fork_started:
                            fork_terminal = True
                            yield ForkSkippedEvent(
                                "main stream completed before the fork trigger"
                            )
                    else:
                        snapshot = snapshot.append(chunk)
                        yield MainChunkEvent(chunk, snapshot)

                        if not fork_started and self.trigger(snapshot, chunk):
                            fork_started = True
                            try:
                                fork_request = await self.builder.build(request, snapshot)
                                validate_request(self.fork_engine, fork_request)
                                decoder = self.decoder_factory()
                                fork_iterator = self.fork_engine.stream(
                                    fork_request
                                ).__aiter__()
                                fork_task = asyncio.create_task(anext(fork_iterator))
                            except Exception as error:
                                fork_terminal = True
                                failure = ForkFailedEvent("build", error)
                                yield failure
                                if self.strict_fork_errors:
                                    raise
                            else:
                                yield ForkStartedEvent(fork_request, snapshot)

                        main_task = asyncio.create_task(anext(main_iterator))

                if fork_task is not None and fork_task in done:
                    try:
                        chunk = fork_task.result()
                    except StopAsyncIteration:
                        fork_task = None
                        try:
                            assert decoder is not None
                            final_calls = tuple(decoder.finish())
                        except Exception as error:
                            fork_terminal = True
                            failure = ForkFailedEvent("decode", error)
                            yield failure
                            if self.strict_fork_errors:
                                raise
                        else:
                            for tool_call in final_calls:
                                decoded.append(tool_call)
                                yield ToolCallEvent(tool_call)
                            fork_terminal = True
                            yield ForkCompletedEvent(tuple(decoded))
                    except Exception as error:
                        fork_task = None
                        fork_terminal = True
                        failure = ForkFailedEvent("stream", error)
                        yield failure
                        if self.strict_fork_errors:
                            raise
                    else:
                        yield ForkChunkEvent(chunk)
                        try:
                            assert decoder is not None
                            calls = tuple(decoder.feed(chunk))
                        except Exception as error:
                            fork_task = None
                            fork_terminal = True
                            failure = ForkFailedEvent("decode", error)
                            yield failure
                            if self.strict_fork_errors:
                                raise
                            if fork_iterator is not None:
                                await _close(fork_iterator)
                        else:
                            for tool_call in calls:
                                decoded.append(tool_call)
                                yield ToolCallEvent(tool_call)
                            assert fork_iterator is not None
                            fork_task = asyncio.create_task(anext(fork_iterator))

            if fork_started and not fork_terminal:
                # Defensive guard for custom iterators with unusual behavior.
                yield ForkCompletedEvent(tuple(decoded))
        finally:
            pending = [task for task in (main_task, fork_task) if task is not None]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await _close(main_iterator)
            if fork_iterator is not None:
                await _close(fork_iterator)

    async def run(self, request: InferenceRequest) -> ForkRunResult:
        main = StreamSnapshot()
        fork_request: InferenceRequest | None = None
        fork_text = ""
        decoded_calls: list[ToolCall] = []
        skipped_reason: str | None = None
        failure: ForkFailedEvent | None = None

        async for event in self.stream(request):
            if isinstance(event, MainChunkEvent):
                main = event.snapshot
            elif isinstance(event, MainCompletedEvent):
                main = event.snapshot
            elif isinstance(event, ForkStartedEvent):
                fork_request = event.request
            elif isinstance(event, ForkChunkEvent):
                fork_text += event.chunk.generated_text
            elif isinstance(event, ToolCallEvent):
                decoded_calls.append(event.tool_call)
            elif isinstance(event, ForkCompletedEvent):
                decoded_calls = list(event.tool_calls)
            elif isinstance(event, ForkSkippedEvent):
                skipped_reason = event.reason
            elif isinstance(event, ForkFailedEvent):
                failure = event

        return ForkRunResult(
            main=main,
            fork_request=fork_request,
            fork_text=fork_text,
            tool_calls=tuple(decoded_calls),
            skipped_reason=skipped_reason,
            failure=failure,
        )

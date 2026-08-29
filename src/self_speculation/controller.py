"""Async orchestration for one main stream and one speculative fork."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from .decoding import DecoderFactory, ToolCallDecoder
from .drafts import DraftBuilder, DraftFeedback, DraftReceipt, DraftRequest
from .engines import InferenceEngine, validate_request
from .events import (
    DraftClearedEvent,
    DraftFailedEvent,
    DraftSubmittedEvent,
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
    draft_receipt: DraftReceipt | None = None
    draft_failure: DraftFailedEvent | None = None


class _DraftOperationError(Exception):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.error = error


async def _close(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


class ForkController:
    """Fork one inference stream, decode it, and optionally feed back a draft.

    Main-stream exceptions remain fatal. Fork and draft failures are best-effort
    by default: they emit typed failure events while the main stream continues
    unchanged. Strict flags surface the corresponding error after its event.
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
        draft_feedback: DraftFeedback | None = None,
        draft_builder: DraftBuilder | None = None,
        strict_draft_errors: bool = False,
    ) -> None:
        if (draft_feedback is None) != (draft_builder is None):
            raise ValueError("draft_feedback and draft_builder must be provided together")
        self.engine = engine
        self.fork_engine = fork_engine or engine
        self.builder = builder
        self.decoder_factory = decoder_factory
        self.trigger = trigger
        self.strict_fork_errors = strict_fork_errors
        self.draft_feedback = draft_feedback
        self.draft_builder = draft_builder
        self.strict_draft_errors = strict_draft_errors

    async def _submit_draft(
        self,
        tool_calls: tuple[ToolCall, ...],
        main_request: InferenceRequest,
        fork_request: InferenceRequest,
    ) -> tuple[DraftRequest, DraftReceipt]:
        assert self.draft_builder is not None
        assert self.draft_feedback is not None
        try:
            draft = await self.draft_builder.build(
                tool_calls, main_request, fork_request
            )
        except Exception as error:
            raise _DraftOperationError("build", error) from error
        try:
            receipt = await self.draft_feedback.submit(draft)
        except Exception as error:
            raise _DraftOperationError("submit", error) from error
        if not isinstance(receipt, DraftReceipt):
            error = TypeError("draft feedback must return DraftReceipt")
            raise _DraftOperationError("submit", error)
        return draft, receipt

    async def stream(self, request: InferenceRequest) -> AsyncIterator[ForkEvent]:
        validate_request(self.engine, request)

        main_iterator = self.engine.stream(request).__aiter__()
        fork_iterator: AsyncIterator[StreamChunk] | None = None
        main_task: asyncio.Task[StreamChunk] | None = asyncio.create_task(
            anext(main_iterator)
        )
        fork_task: asyncio.Task[StreamChunk] | None = None
        draft_task: asyncio.Task[tuple[DraftRequest, DraftReceipt]] | None = None
        snapshot = StreamSnapshot()
        fork_started = False
        fork_terminal = False
        fork_request: InferenceRequest | None = None
        decoder: ToolCallDecoder | None = None
        decoded: list[ToolCall] = []
        draft_started = False
        draft_cleared = False

        def start_draft(tool_calls: tuple[ToolCall, ...]) -> None:
            nonlocal draft_started, draft_task
            if (
                draft_started
                or not tool_calls
                or self.draft_feedback is None
                or self.draft_builder is None
                or fork_request is None
            ):
                return
            draft_started = True
            draft_task = asyncio.create_task(
                self._submit_draft(tool_calls, request, fork_request)
            )

        try:
            while any(task is not None for task in (main_task, fork_task, draft_task)):
                tasks = {
                    task
                    for task in (main_task, fork_task, draft_task)
                    if task is not None
                }
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Process main first when tasks complete in the same loop. This
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
                            start_draft(tuple(decoded))
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

                if draft_task is not None and draft_task in done:
                    completed_draft_task = draft_task
                    draft_task = None
                    try:
                        draft, receipt = completed_draft_task.result()
                    except _DraftOperationError as operation_error:
                        failure = DraftFailedEvent(
                            operation_error.stage, operation_error.error
                        )
                        yield failure
                        if self.strict_draft_errors:
                            raise operation_error.error
                    else:
                        yield DraftSubmittedEvent(draft, receipt)

            if fork_started and not fork_terminal:
                yield ForkCompletedEvent(tuple(decoded))

            if draft_started and self.draft_feedback is not None:
                try:
                    await self.draft_feedback.clear(request.request_id)
                except Exception as error:
                    failure = DraftFailedEvent("clear", error)
                    yield failure
                    if self.strict_draft_errors:
                        raise
                else:
                    yield DraftClearedEvent(request.request_id)
                finally:
                    draft_cleared = True
        finally:
            pending = [
                task
                for task in (main_task, fork_task, draft_task)
                if task is not None
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await _close(main_iterator)
            if fork_iterator is not None:
                await _close(fork_iterator)
            if (
                draft_started
                and not draft_cleared
                and self.draft_feedback is not None
            ):
                try:
                    await self.draft_feedback.clear(request.request_id)
                except Exception:
                    pass

    async def run(self, request: InferenceRequest) -> ForkRunResult:
        main = StreamSnapshot()
        fork_request: InferenceRequest | None = None
        fork_text = ""
        decoded_calls: list[ToolCall] = []
        skipped_reason: str | None = None
        failure: ForkFailedEvent | None = None
        draft_receipt: DraftReceipt | None = None
        draft_failure: DraftFailedEvent | None = None

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
            elif isinstance(event, DraftSubmittedEvent):
                draft_receipt = event.receipt
            elif isinstance(event, DraftFailedEvent):
                draft_failure = event

        return ForkRunResult(
            main=main,
            fork_request=fork_request,
            fork_text=fork_text,
            tool_calls=tuple(decoded_calls),
            skipped_reason=skipped_reason,
            failure=failure,
            draft_receipt=draft_receipt,
            draft_failure=draft_failure,
        )

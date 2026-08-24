from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator, Iterable

from self_speculation import (
    EngineCapabilities,
    ForkController,
    ForkFailedEvent,
    ForkSkippedEvent,
    ForkStartedEvent,
    InferenceRequest,
    MainChunkEvent,
    PrefixForkBuilder,
    StreamChunk,
    ToolCall,
    ToolCallEvent,
)


class FakeDecoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.buffer = ""
        self.emitted = False

    def feed(self, chunk: StreamChunk) -> Iterable[ToolCall]:
        if self.fail:
            raise ValueError("cannot decode")
        self.buffer += chunk.generated_text
        if "search" in self.buffer and not self.emitted:
            self.emitted = True
            return (ToolCall("search", {"q": "spork"}, format="fake"),)
        return ()

    def finish(self) -> Iterable[ToolCall]:
        return ()


class CoordinatedEngine:
    name = "coordinated"
    capabilities = EngineCapabilities(prompt=True)

    def __init__(self) -> None:
        self.fork_entered = asyncio.Event()
        self.seen_requests: list[InferenceRequest] = []

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        self.seen_requests.append(request)
        if request.request_id.endswith(":fork"):
            self.fork_entered.set()
            yield StreamChunk(text='search({"q":"spork"})')
            return

        yield StreamChunk(text="first", token_ids=(1,))
        await asyncio.wait_for(self.fork_entered.wait(), timeout=1)
        yield StreamChunk(text="second", token_ids=(2,), finish_reason="stop")


class EmptyEngine:
    name = "empty"
    capabilities = EngineCapabilities(prompt=True)

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(finish_reason="stop")


class ForkControllerTest(unittest.IsolatedAsyncioTestCase):
    async def test_forks_after_first_output_and_keeps_main_streaming(self) -> None:
        engine = CoordinatedEngine()
        controller = ForkController(
            engine,
            PrefixForkBuilder(forced_prefix="<tool>"),
            FakeDecoder,
        )

        events = [
            event
            async for event in controller.stream(
                InferenceRequest(prompt="PROMPT", request_id="turn")
            )
        ]

        first_main = next(event for event in events if isinstance(event, MainChunkEvent))
        started = next(event for event in events if isinstance(event, ForkStartedEvent))
        decoded = next(event for event in events if isinstance(event, ToolCallEvent))
        self.assertEqual(first_main.snapshot.generated_text, "first")
        self.assertEqual(started.request.prompt, "PROMPTfirst<tool>")
        self.assertEqual(decoded.tool_call.name, "search")
        self.assertEqual(
            [request.request_id for request in engine.seen_requests],
            ["turn", "turn:fork"],
        )

    async def test_run_collects_both_streams(self) -> None:
        result = await ForkController(
            CoordinatedEngine(),
            PrefixForkBuilder(forced_prefix="<tool>"),
            FakeDecoder,
        ).run(InferenceRequest(prompt="P", request_id="run"))

        self.assertEqual(result.main.generated_text, "firstsecond")
        self.assertIn("search", result.fork_text)
        self.assertEqual([call.name for call in result.tool_calls], ["search"])
        self.assertIsNone(result.failure)

    async def test_skips_when_main_finishes_without_generated_output(self) -> None:
        controller = ForkController(
            EmptyEngine(), PrefixForkBuilder(forced_prefix="x"), FakeDecoder
        )
        events = [
            event
            async for event in controller.stream(InferenceRequest(prompt="P"))
        ]
        self.assertTrue(any(isinstance(event, ForkSkippedEvent) for event in events))

    async def test_decode_failure_is_nonfatal_by_default(self) -> None:
        controller = ForkController(
            CoordinatedEngine(),
            PrefixForkBuilder(forced_prefix="x"),
            lambda: FakeDecoder(fail=True),
        )

        result = await controller.run(InferenceRequest(prompt="P", request_id="safe"))

        self.assertEqual(result.main.generated_text, "firstsecond")
        self.assertIsNotNone(result.failure)
        assert result.failure is not None
        self.assertEqual(result.failure.stage, "decode")

    async def test_strict_mode_surfaces_decode_failure(self) -> None:
        controller = ForkController(
            CoordinatedEngine(),
            PrefixForkBuilder(forced_prefix="x"),
            lambda: FakeDecoder(fail=True),
            strict_fork_errors=True,
        )

        with self.assertRaisesRegex(ValueError, "cannot decode"):
            await controller.run(InferenceRequest(prompt="P", request_id="strict"))


if __name__ == "__main__":
    unittest.main()

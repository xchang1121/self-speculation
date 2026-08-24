from __future__ import annotations

import unittest
from collections.abc import AsyncIterator

from self_speculation import (
    EngineCapabilities,
    EngineCapabilityError,
    InferenceEngine,
    InferenceRequest,
    StreamChunk,
    StreamSnapshot,
    ToolCallDelta,
    validate_request,
)


class FakeEngine:
    name = "fake"
    capabilities = EngineCapabilities(prompt=True, chat=False)

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text=request.prompt or "")


class InferenceRequestTest(unittest.TestCase):
    def test_normalizes_sequence_inputs_and_copies_with_validation(self) -> None:
        request = InferenceRequest(
            prompt="hello",
            tools=[{"type": "function"}],
            stop=["END"],
            max_tokens=8,
        )

        self.assertEqual(request.input_mode, "prompt")
        self.assertIsInstance(request.tools, tuple)
        self.assertIsInstance(request.stop, tuple)
        self.assertEqual(request.with_changes(max_tokens=4).max_tokens, 4)

    def test_requires_exactly_one_input_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            InferenceRequest()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            InferenceRequest(prompt="x", messages=({"role": "user"},))

    def test_rejects_invalid_limits_and_stops(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            InferenceRequest(prompt="x", max_tokens=0)
        with self.assertRaisesRegex(ValueError, "stop strings"):
            InferenceRequest(prompt="x", stop=("",))


class StreamModelTest(unittest.TestCase):
    def test_snapshot_preserves_reasoning_and_content(self) -> None:
        first = StreamChunk(reasoning="think", token_ids=(1,))
        second = StreamChunk(
            text="answer",
            tool_call_deltas=(ToolCallDelta(name="search"),),
            token_ids=(2, 3),
            finish_reason="tool_calls",
        )

        snapshot = StreamSnapshot().append(first).append(second)

        self.assertEqual(snapshot.generated_text, "thinkanswer")
        self.assertEqual(snapshot.reasoning, "think")
        self.assertEqual(snapshot.content, "answer")
        self.assertEqual(snapshot.chunk_count, 2)
        self.assertEqual(snapshot.output_chunk_count, 2)
        self.assertEqual(snapshot.token_count, 3)
        self.assertEqual(snapshot.finish_reason, "tool_calls")


class EngineProtocolTest(unittest.TestCase):
    def test_runtime_protocol_and_capability_validation(self) -> None:
        engine = FakeEngine()
        self.assertIsInstance(engine, InferenceEngine)
        validate_request(engine, InferenceRequest(prompt="ok"))
        with self.assertRaises(EngineCapabilityError):
            validate_request(engine, InferenceRequest(messages=({"role": "user"},)))


if __name__ == "__main__":
    unittest.main()

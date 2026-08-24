from __future__ import annotations

import unittest

from self_speculation import (
    CallableEngine,
    InferenceRequest,
    StreamChunk,
    default_chunk_mapper,
)


async def collect(engine: CallableEngine) -> list[StreamChunk]:
    return [chunk async for chunk in engine.stream(InferenceRequest(prompt="x"))]


class CallableEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_async_generator_and_default_item_shapes(self) -> None:
        async def factory(request: InferenceRequest):
            yield "hello"
            yield {
                "reasoning": "think",
                "tool_call_deltas": [
                    {"index": 0, "id": "c1", "name": "search", "arguments": "{}"}
                ],
                "token_ids": [1, 2],
                "logprobs": [
                    {"token": "x", "logprob": -0.1, "top_logprobs": {"x": -0.1}}
                ],
            }

        chunks = await collect(CallableEngine(factory))

        self.assertEqual(chunks[0].text, "hello")
        self.assertEqual(chunks[1].reasoning, "think")
        self.assertEqual(chunks[1].tool_call_deltas[0].call_id, "c1")
        self.assertEqual(chunks[1].token_ids, (1, 2))

    async def test_adapts_awaitable_factory_and_async_mapper(self) -> None:
        async def factory(request: InferenceRequest):
            return [1, 2]

        async def mapper(value: int) -> StreamChunk:
            return StreamChunk(text=str(value * 2))

        chunks = await collect(CallableEngine(factory, mapper))
        self.assertEqual([chunk.text for chunk in chunks], ["2", "4"])

    async def test_advances_synchronous_generator_and_closes_it(self) -> None:
        closed: list[bool] = []

        def factory(request: InferenceRequest):
            try:
                yield "a"
                yield "b"
            finally:
                closed.append(True)

        chunks = await collect(CallableEngine(factory))
        self.assertEqual([chunk.text for chunk in chunks], ["a", "b"])
        self.assertEqual(closed, [True])

    async def test_rejects_invalid_source_or_mapper_output(self) -> None:
        invalid_source = CallableEngine(lambda request: 123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            await collect(invalid_source)

        invalid_mapper = CallableEngine(
            lambda request: [1], lambda value: "wrong"  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(TypeError, "mapper"):
            await collect(invalid_mapper)

    def test_default_mapper_preserves_raw_mapping(self) -> None:
        raw = {"text": "x", "finish_reason": "stop", "usage": {"output": 1}}
        chunk = default_chunk_mapper(raw)
        self.assertIs(chunk.raw, raw)
        self.assertEqual(chunk.finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()

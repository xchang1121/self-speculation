from __future__ import annotations

import unittest
from types import SimpleNamespace

from self_speculation import InferenceRequest, VLLMNativeEngine, fit_request_to_context


def output(
    text: str,
    token_ids: list[int],
    *,
    finished: bool = False,
    finish_reason: str | None = None,
):
    item = SimpleNamespace(
        text=text,
        token_ids=token_ids,
        finish_reason=finish_reason,
        logprobs=None,
    )
    return SimpleNamespace(
        outputs=[item],
        finished=finished,
        prompt_token_ids=[10, 11],
    )


class FakeVLLM:
    def __init__(self, items) -> None:
        self.items = items
        self.calls = []
        self.aborted = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)

        async def results():
            for item in self.items:
                yield item

        return results()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class VLLMNativeEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_counts_pre_tokenized_native_prompts_for_context_budgeting(self) -> None:
        engine = VLLMNativeEngine(
            FakeVLLM([]),
            sampling_params_factory=lambda request: object(),
            prompt_renderer=lambda request: {"prompt_token_ids": [1, 2, 3]},
            max_context_tokens=5,
        )

        bounded, _ = await fit_request_to_context(
            engine,
            InferenceRequest(messages=({"role": "user", "content": "x"},), max_tokens=9),
        )

        self.assertEqual(bounded.max_tokens, 2)

    async def test_normalizes_cumulative_outputs(self) -> None:
        native = FakeVLLM(
            [
                output("a", [1]),
                output("abc", [1, 2, 3], finished=True, finish_reason="length"),
            ]
        )
        engine = VLLMNativeEngine(
            native,
            sampling_params_factory=lambda request: {"max_tokens": 3},
            output_mode="cumulative",
        )
        chunks = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(prompt="P", request_id="native-1")
            )
        ]

        self.assertEqual([chunk.text for chunk in chunks], ["a", "bc"])
        self.assertEqual([chunk.token_ids for chunk in chunks], [(1,), (2, 3)])
        self.assertEqual(chunks[-1].finish_reason, "length")
        self.assertEqual(chunks[-1].usage["completion_tokens"], 3)
        self.assertEqual(native.calls[0]["prompt"], "P")
        self.assertEqual(native.aborted, [])

    async def test_normalizes_delta_outputs_and_passes_generate_kwargs(self) -> None:
        native = FakeVLLM(
            [output("a", [1]), output("b", [2], finished=True)]
        )
        engine = VLLMNativeEngine(
            native,
            sampling_params_factory=lambda request: object(),
            output_mode="delta",
        )
        request = InferenceRequest(
            prompt="P",
            request_id="native-2",
            extra={"generate_kwargs": {"priority": 4}},
        )
        chunks = [chunk async for chunk in engine.stream(request)]
        self.assertEqual([chunk.text for chunk in chunks], ["a", "b"])
        self.assertEqual(chunks[-1].finish_reason, "stop")
        self.assertEqual(native.calls[0]["priority"], 4)

    async def test_supports_async_chat_renderer(self) -> None:
        native = FakeVLLM([output("ok", [1], finished=True)])

        async def render(request: InferenceRequest):
            return {"prompt": "rendered", "prompt_token_ids": [9]}

        engine = VLLMNativeEngine(
            native,
            sampling_params_factory=lambda request: object(),
            prompt_renderer=render,
            output_mode="delta",
        )
        _ = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(messages=({"role": "user"},), request_id="chat")
            )
        ]
        self.assertEqual(native.calls[0]["prompt"]["prompt"], "rendered")

    async def test_aborts_when_consumer_closes_before_completion(self) -> None:
        native = FakeVLLM([output("a", [1]), output("b", [2])])
        engine = VLLMNativeEngine(
            native,
            sampling_params_factory=lambda request: object(),
            output_mode="delta",
        )
        stream = engine.stream(InferenceRequest(prompt="P", request_id="cancel-me"))
        first = await anext(stream)
        self.assertEqual(first.text, "a")
        await stream.aclose()
        self.assertEqual(native.aborted, ["cancel-me"])

    async def test_does_not_abort_a_naturally_exhausted_legacy_stream(self) -> None:
        native = FakeVLLM([output("a", [1])])
        engine = VLLMNativeEngine(
            native,
            sampling_params_factory=lambda request: object(),
            output_mode="delta",
        )
        _ = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(prompt="P", request_id="legacy")
            )
        ]
        self.assertEqual(native.aborted, [])


if __name__ == "__main__":
    unittest.main()

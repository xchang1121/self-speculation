from __future__ import annotations

import unittest

from self_speculation import (
    CallableForkBuilder,
    ForkBuildError,
    InferenceRequest,
    PrefixForkBuilder,
    StreamSnapshot,
)


class PrefixForkBuilderTest(unittest.IsolatedAsyncioTestCase):
    async def test_builds_prefix_fork_from_raw_main_request(self) -> None:
        main = InferenceRequest(
            prompt="PROMPT",
            model="model-a",
            request_id="turn-1",
            extra={"main-only": True},
        )
        snapshot = StreamSnapshot(generated_text="thinking")
        builder = PrefixForkBuilder(
            forced_prefix="</think><tool_call>",
            stop=("</tool_call>",),
            max_tokens=64,
            extra={"logprobs": 5},
        )

        fork = await builder.build(main, snapshot)

        self.assertEqual(fork.prompt, "PROMPTthinking</think><tool_call>")
        self.assertEqual(fork.model, "model-a")
        self.assertEqual(fork.request_id, "turn-1:fork")
        self.assertEqual(fork.stop, ("</tool_call>",))
        self.assertEqual(fork.extra, {"logprobs": 5})

    async def test_can_render_chat_prompt_asynchronously(self) -> None:
        main = InferenceRequest(
            messages=({"role": "user", "content": "hello"},),
            request_id="chat-1",
        )

        async def render(request: InferenceRequest) -> str:
            return "CHAT_TEMPLATE"

        fork = await PrefixForkBuilder(
            forced_prefix="<tool_call>", prompt_renderer=render
        ).build(main, StreamSnapshot(generated_text="x"))

        self.assertEqual(fork.prompt, "CHAT_TEMPLATEx<tool_call>")

    async def test_chat_requires_a_renderer(self) -> None:
        main = InferenceRequest(messages=({"role": "user"},))
        with self.assertRaises(ForkBuildError):
            await PrefixForkBuilder(forced_prefix="tool:").build(
                main, StreamSnapshot()
            )


class CallableForkBuilderTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_custom_request_factory(self) -> None:
        main = InferenceRequest(prompt="main", request_id="r")

        async def factory(
            request: InferenceRequest, snapshot: StreamSnapshot
        ) -> InferenceRequest:
            return InferenceRequest(
                messages=({"role": "assistant", "content": snapshot.content},),
                request_id=request.request_id + ":custom",
            )

        fork = await CallableForkBuilder(factory).build(
            main, StreamSnapshot(content="partial")
        )

        self.assertEqual(fork.input_mode, "chat")
        self.assertEqual(fork.request_id, "r:custom")

    async def test_rejects_wrong_factory_result(self) -> None:
        builder = CallableForkBuilder(lambda request, snapshot: "not a request")
        with self.assertRaises(ForkBuildError):
            await builder.build(InferenceRequest(prompt="x"), StreamSnapshot())


if __name__ == "__main__":
    unittest.main()

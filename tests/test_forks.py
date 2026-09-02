from __future__ import annotations

import unittest

from self_speculation import (
    CallableForkBuilder,
    ContinuationFormatError,
    ContinuationPlanner,
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
        self.assertEqual(fork.parent_request_id, "turn-1")
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
        self.assertEqual(fork.parent_request_id, "r")

    async def test_rejects_wrong_factory_result(self) -> None:
        builder = CallableForkBuilder(lambda request, snapshot: "not a request")
        with self.assertRaises(ForkBuildError):
            await builder.build(InferenceRequest(prompt="x"), StreamSnapshot())


class ContinuationPlannerTest(unittest.TestCase):
    def test_uses_each_formats_name_aligned_probe_prefix(self) -> None:
        prefixes = {
            "tagged_json": '<tool_call>\n{"name":"',
            "qwen_xml": "<tool_call>\n<function=",
            "deepseek_dsml": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="',
            "deepseek_v3": (
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                "function<｜tool▁sep｜>"
            ),
            "mistral_json": "[TOOL_CALLS]",
            "llama_json": "<|python_tag|>",
            "pythonic": "<|python_tag|>",
        }

        for format_name, prefix in prefixes.items():
            with self.subTest(format_name=format_name):
                plan = ContinuationPlanner().plan(
                    "prompt",
                    StreamSnapshot(generated_text="text", content="text"),
                    tool_format=format_name,
                )
                self.assertEqual(plan.forced_suffix, prefix)

    def test_closes_the_envelope_that_is_actually_open(self) -> None:
        plan = ContinuationPlanner().plan(
            "<|assistant|><analysis>\n",
            StreamSnapshot(generated_text="inspect", content="inspect"),
            tool_format="tagged_json",
        )

        self.assertEqual(plan.observed_text, "inspect")
        self.assertEqual(
            plan.forced_suffix,
            '\n</analysis>\n\n<tool_call>\n{"name":"',
        )
        self.assertEqual(plan.reasoning_format, "analysis_xml")

    def test_restores_a_hidden_qwen_transition_before_visible_content(self) -> None:
        plan = ContinuationPlanner().plan(
            "<|im_start|>assistant\n<think>\n",
            StreamSnapshot(
                generated_text="reasonanswer",
                reasoning="reason",
                content="answer",
            ),
            tool_format="qwen_xml",
        )

        self.assertEqual(
            plan.observed_text,
            "reason\n</think>\n\nanswer",
        )
        self.assertEqual(plan.forced_suffix, "<tool_call>\n<function=")
        self.assertTrue(plan.reconstructed_transition)

    def test_rejects_opaque_reasoning_without_a_text_envelope(self) -> None:
        with self.assertRaisesRegex(
            ContinuationFormatError,
            "engine-native fork",
        ):
            ContinuationPlanner().plan(
                "structured-provider-prompt",
                StreamSnapshot(generated_text="summary", reasoning="summary"),
                tool_format="tagged_json",
            )

    def test_rejects_a_tool_prefix_from_another_format(self) -> None:
        with self.assertRaisesRegex(ContinuationFormatError, "deepseek_dsml"):
            ContinuationPlanner().plan(
                "prompt",
                StreamSnapshot(generated_text="text", content="text"),
                tool_format="deepseek_dsml",
                forced_prefix="<tool_call>",
            )


if __name__ == "__main__":
    unittest.main()

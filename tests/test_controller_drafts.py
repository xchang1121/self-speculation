from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator

from self_speculation import (
    DraftFailedEvent,
    DraftReceipt,
    DraftRequest,
    EngineCapabilities,
    ForkController,
    InferenceRequest,
    PrefixForkBuilder,
    StreamChunk,
    ToolCallDraftBuilder,
    default_decoder,
    default_draft_boundary,
    format_tool_call_draft,
)


class RecordingFeedback:
    name = "recording"

    def __init__(self, *, fail_submit: bool = False, fail_clear: bool = False) -> None:
        self.fail_submit = fail_submit
        self.fail_clear = fail_clear
        self.submitted = asyncio.Event()
        self.drafts: list[DraftRequest] = []
        self.cleared: list[str] = []
        self.main_finished = False

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        self.drafts.append(draft)
        self.submitted.set()
        if self.fail_submit:
            raise RuntimeError("submit failed")
        return DraftReceipt(
            request_id=draft.request_id,
            registered=True,
            draft_token_count=len(draft.token_ids),
        )

    async def clear(self, request_id: str) -> None:
        self.cleared.append(request_id)
        if self.fail_clear:
            raise RuntimeError("clear failed")
        if not self.main_finished:
            raise AssertionError("draft was cleared before the main stream finished")


class DraftAwareEngine:
    name = "draft-aware-fake"
    capabilities = EngineCapabilities(prompt=True)

    def __init__(self, feedback: RecordingFeedback) -> None:
        self.feedback = feedback

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        if request.request_id.endswith(":fork"):
            yield StreamChunk(
                text=(
                    '<tool_call>{"name":"search",'
                    '"arguments":{"q":"spork"}}</tool_call>'
                )
            )
            return

        yield StreamChunk(text="first", token_ids=(1,))
        await asyncio.wait_for(self.feedback.submitted.wait(), timeout=1)
        yield StreamChunk(text="second", token_ids=(2,), finish_reason="stop")
        self.feedback.main_finished = True


def controller(
    feedback: RecordingFeedback,
    *,
    strict: bool = False,
) -> ForkController:
    return ForkController(
        DraftAwareEngine(feedback),
        PrefixForkBuilder(forced_prefix="<tool_call>"),
        lambda: default_decoder("tagged_json"),
        draft_feedback=feedback,
        draft_builder=ToolCallDraftBuilder(
            formatter=format_tool_call_draft,
            tokenizer=lambda text: list(text.encode("utf-8")),
            boundary_resolver=default_draft_boundary,
        ),
        strict_draft_errors=strict,
    )


class ControllerDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_submits_decoded_action_before_main_finishes_then_clears(self) -> None:
        feedback = RecordingFeedback()

        result = await controller(feedback).run(
            InferenceRequest(prompt="P", request_id="turn")
        )

        self.assertEqual(result.main.generated_text, "firstsecond")
        self.assertEqual(len(feedback.drafts), 1)
        draft = feedback.drafts[0]
        self.assertEqual(draft.request_id, "turn")
        self.assertIn('"name":"search"', draft.text)
        self.assertTrue(draft.token_ids)
        self.assertEqual(draft.boundary.text, "<tool_call>")
        self.assertIsNotNone(result.draft_receipt)
        self.assertIsNone(result.draft_failure)
        self.assertEqual(feedback.cleared, ["turn"])

    async def test_submit_failure_is_best_effort_and_still_clears(self) -> None:
        feedback = RecordingFeedback(fail_submit=True)

        result = await controller(feedback).run(
            InferenceRequest(prompt="P", request_id="safe")
        )

        self.assertEqual(result.main.generated_text, "firstsecond")
        self.assertIsInstance(result.draft_failure, DraftFailedEvent)
        assert result.draft_failure is not None
        self.assertEqual(result.draft_failure.stage, "submit")
        self.assertEqual(feedback.cleared, ["safe"])

    async def test_clear_failure_is_reported(self) -> None:
        feedback = RecordingFeedback(fail_clear=True)
        result = await controller(feedback).run(
            InferenceRequest(prompt="P", request_id="clear")
        )
        self.assertIsNotNone(result.draft_failure)
        assert result.draft_failure is not None
        self.assertEqual(result.draft_failure.stage, "clear")

    async def test_strict_submit_failure_surfaces(self) -> None:
        feedback = RecordingFeedback(fail_submit=True)
        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            await controller(feedback, strict=True).run(
                InferenceRequest(prompt="P", request_id="strict")
            )
        self.assertEqual(feedback.cleared, ["strict"])

    def test_requires_feedback_and_builder_as_a_pair(self) -> None:
        feedback = RecordingFeedback()
        with self.assertRaisesRegex(ValueError, "provided together"):
            ForkController(
                DraftAwareEngine(feedback),
                PrefixForkBuilder(forced_prefix="x"),
                lambda: default_decoder("tagged_json"),
                draft_feedback=feedback,
            )


if __name__ == "__main__":
    unittest.main()

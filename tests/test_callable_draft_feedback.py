from __future__ import annotations

import asyncio
import threading
import unittest

from self_speculation import (
    CallableDraftFeedback,
    DraftBundle,
    DraftReceipt,
    DraftRequest,
)


def bundle(draft: DraftRequest) -> DraftBundle:
    return DraftBundle(draft.request_id, (draft,))


class CallableDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_sync_callbacks_without_blocking_the_event_loop(self) -> None:
        callback_thread_ids: list[int] = []
        cleared: list[str] = []

        def submit(batch: DraftBundle):
            draft = batch.drafts[0]
            callback_thread_ids.append(threading.get_ident())
            return {
                "status": "ok",
                "request_id": draft.request_id,
                "n_tokens": len(draft.token_ids),
            }

        feedback = CallableDraftFeedback(submit, cleared.append)
        draft = DraftRequest(request_id="req", token_ids=(1, 2, 3))
        receipt = await feedback.submit(bundle(draft))
        await feedback.clear("req")

        self.assertNotEqual(callback_thread_ids, [threading.get_ident()])
        self.assertTrue(receipt.registered)
        self.assertEqual(receipt.draft_token_count, 3)
        self.assertEqual(receipt.details["status"], "ok")
        self.assertEqual(cleared, ["req"])

    async def test_adapts_async_callbacks_and_native_receipts(self) -> None:
        async def submit(batch: DraftBundle) -> DraftReceipt:
            await asyncio.sleep(0)
            return DraftReceipt(batch.request_id, True, accepted_token_count=2)

        feedback = CallableDraftFeedback(submit)
        receipt = await feedback.submit(
            bundle(DraftRequest(request_id="async", text="draft"))
        )
        await feedback.clear("async")
        self.assertEqual(receipt.accepted_token_count, 2)

    async def test_normalizes_none_bool_and_mapping_failures(self) -> None:
        draft = DraftRequest(request_id="x", token_ids=(7,))
        accepted = await CallableDraftFeedback(lambda item: None).submit(bundle(draft))
        rejected = await CallableDraftFeedback(lambda item: False).submit(bundle(draft))

        self.assertTrue(accepted.registered)
        self.assertFalse(rejected.registered)
        with self.assertRaisesRegex(ValueError, "does not match"):
            await CallableDraftFeedback(
                lambda item: {"request_id": "other"}
            ).submit(bundle(draft))
        with self.assertRaisesRegex(TypeError, "must return"):
            await CallableDraftFeedback(lambda item: object()).submit(bundle(draft))

    async def test_normalizes_clear_verification_metrics(self) -> None:
        feedback = CallableDraftFeedback(
            lambda draft: True,
            lambda request_id: {
                "request_id": request_id,
                "verification": {
                    "num_spec_steps": 2,
                    "num_draft_tokens": 5,
                    "num_accepted_draft_tokens": 3,
                    "num_rejected_draft_tokens": 2,
                    "per_step_drafted": [3, 2],
                    "per_step_accepted": [2, 1],
                },
            },
        )

        outcome = await feedback.clear("verified")

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.proposed_tokens if outcome else None, 5)
        self.assertEqual(outcome.accepted_tokens if outcome else None, 3)
        self.assertEqual(outcome.rejected_tokens if outcome else None, 2)


if __name__ == "__main__":
    unittest.main()

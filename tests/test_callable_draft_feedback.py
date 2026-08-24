from __future__ import annotations

import asyncio
import threading
import unittest

from self_speculation import CallableDraftFeedback, DraftReceipt, DraftRequest


class CallableDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_sync_callbacks_without_blocking_the_event_loop(self) -> None:
        callback_thread_ids: list[int] = []
        cleared: list[str] = []

        def submit(draft: DraftRequest):
            callback_thread_ids.append(threading.get_ident())
            return {
                "status": "ok",
                "request_id": draft.request_id,
                "n_tokens": len(draft.token_ids),
            }

        feedback = CallableDraftFeedback(submit, cleared.append)
        draft = DraftRequest(request_id="req", token_ids=(1, 2, 3))
        receipt = await feedback.submit(draft)
        await feedback.clear("req")

        self.assertNotEqual(callback_thread_ids, [threading.get_ident()])
        self.assertTrue(receipt.registered)
        self.assertEqual(receipt.draft_token_count, 3)
        self.assertEqual(receipt.details["status"], "ok")
        self.assertEqual(cleared, ["req"])

    async def test_adapts_async_callbacks_and_native_receipts(self) -> None:
        async def submit(draft: DraftRequest) -> DraftReceipt:
            await asyncio.sleep(0)
            return DraftReceipt(draft.request_id, True, accepted_token_count=2)

        feedback = CallableDraftFeedback(submit)
        receipt = await feedback.submit(
            DraftRequest(request_id="async", text="draft")
        )
        await feedback.clear("async")
        self.assertEqual(receipt.accepted_token_count, 2)

    async def test_normalizes_none_bool_and_mapping_failures(self) -> None:
        draft = DraftRequest(request_id="x", token_ids=(7,))
        accepted = await CallableDraftFeedback(lambda item: None).submit(draft)
        rejected = await CallableDraftFeedback(lambda item: False).submit(draft)

        self.assertTrue(accepted.registered)
        self.assertFalse(rejected.registered)
        with self.assertRaisesRegex(ValueError, "does not match"):
            await CallableDraftFeedback(
                lambda item: {"request_id": "other"}
            ).submit(draft)
        with self.assertRaisesRegex(TypeError, "must return"):
            await CallableDraftFeedback(lambda item: object()).submit(draft)


if __name__ == "__main__":
    unittest.main()

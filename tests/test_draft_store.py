from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from self_speculation import (
    BoundaryDraftFeedback,
    BoundaryDraftStore,
    DraftBoundary,
    DraftRequest,
)


def draft(
    request_id: str,
    *,
    tokens: tuple[int, ...] = (20, 21, 22),
    boundary: tuple[int, ...] = (10, 11),
    prompt_tokens: int = 2,
) -> DraftRequest:
    return DraftRequest(
        request_id=request_id,
        token_ids=tokens,
        boundary=DraftBoundary(token_ids=boundary),
        prompt_token_count=prompt_tokens,
    )


class BoundaryDraftStoreTest(unittest.TestCase):
    def test_finds_multitoken_boundary_and_skips_generated_prefix(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        receipt = store.register(draft("one"))

        self.assertIsNone(store.offer("one", [90, 91, 5, 10]))
        proposal = store.offer("one", [90, 91, 5, 10, 11, 20])

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.token_ids, (21, 22))
        self.assertEqual(proposal.skipped_prefix_tokens, 1)
        self.assertEqual(proposal.boundary_index, 1)
        self.assertEqual(receipt.details["boundary_token_count"], 2)
        self.assertIsNone(store.offer("one", [90, 91, 10, 11, 20]))
        snapshot = store.snapshot()
        self.assertEqual(snapshot.injections, 1)
        self.assertEqual(snapshot.proposed_tokens, 2)

    def test_keeps_stable_request_ids_isolated(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        store.register(draft("alpha", tokens=(1, 2), boundary=(8,)))
        store.register(draft("beta", tokens=(3, 4), boundary=(9,)))

        alpha = store.offer("alpha", [0, 0, 8])
        beta = store.offer("beta", [0, 0, 9])

        self.assertEqual(alpha.token_ids if alpha else (), (1, 2))
        self.assertEqual(beta.token_ids if beta else (), (3, 4))
        self.assertEqual(store.snapshot().active_requests, 2)
        self.assertTrue(store.clear("alpha"))
        self.assertFalse(store.clear("missing"))
        self.assertEqual(store.clear_all(), 1)

    def test_discards_divergent_or_stale_drafts(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8, inject_window=1)
        store.register(draft("diverged"))
        self.assertIsNone(
            store.offer("diverged", [90, 91, 10, 11, 99])
        )
        store.register(draft("stale"))
        self.assertIsNone(store.offer("stale", [90, 91, 10, 11, 20, 21]))

        snapshot = store.snapshot()
        self.assertEqual(snapshot.divergent_drafts, 1)
        self.assertEqual(snapshot.stale_drafts, 1)

    def test_at_most_one_thread_receives_a_proposal(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        store.register(draft("race"))
        sequence = [90, 91, 10, 11]
        with ThreadPoolExecutor(max_workers=8) as executor:
            proposals = list(
                executor.map(lambda _: store.offer("race", sequence), range(32))
            )
        self.assertEqual(sum(item is not None for item in proposals), 1)

    def test_tokenizes_text_boundaries_and_validates_inputs(self) -> None:
        store = BoundaryDraftStore(boundary_tokenizer=lambda text: [7, 8])
        store.register(
            DraftRequest(
                request_id="text",
                token_ids=(1,),
                boundary=DraftBoundary(text="boundary"),
            )
        )
        self.assertEqual(store.offer("text", [7, 8]).token_ids, (1,))

        with self.assertRaisesRegex(ValueError, "requires token_ids"):
            store.register(
                DraftRequest(
                    request_id="raw",
                    text="body",
                    boundary=DraftBoundary(token_ids=(1,)),
                )
            )
        with self.assertRaisesRegex(ValueError, "boundary"):
            BoundaryDraftStore().register(
                DraftRequest(request_id="no-boundary", token_ids=(1,))
            )
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            store.offer("text", [1], sequence_length=2)


class BoundaryDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_store_as_async_feedback(self) -> None:
        store = BoundaryDraftStore()
        feedback = BoundaryDraftFeedback(store)
        receipt = await feedback.submit(draft("async"))
        self.assertTrue(receipt.registered)
        await feedback.clear("async")
        self.assertEqual(store.snapshot().active_requests, 0)


if __name__ == "__main__":
    unittest.main()

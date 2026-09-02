from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from self_speculation import (
    BoundaryDraftFeedback,
    BoundaryDraftStore,
    DraftBundle,
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


def register(store: BoundaryDraftStore, *drafts: DraftRequest):
    return store.register_bundle(DraftBundle(drafts[0].request_id, drafts))


class BoundaryDraftStoreTest(unittest.TestCase):
    def test_finds_multitoken_boundary_and_skips_generated_prefix(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        receipt = register(store, draft("one"))

        self.assertIsNone(store.offer("one", [90, 91, 5, 10]))
        proposal = store.offer("one", [90, 91, 5, 10, 11, 20])

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.token_ids, (21, 22))
        self.assertEqual(proposal.skipped_prefix_tokens, 1)
        self.assertEqual(proposal.boundary_index, 1)
        self.assertEqual(receipt.details["candidate_count"], 1)
        self.assertIsNone(store.offer("one", [90, 91, 10, 11, 20]))
        snapshot = store.snapshot()
        self.assertEqual(snapshot.injections, 1)
        self.assertEqual(snapshot.proposed_tokens, 2)

    def test_keeps_stable_request_ids_isolated(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        register(store, draft("alpha", tokens=(1, 2), boundary=(8,)))
        register(store, draft("beta", tokens=(3, 4), boundary=(9,)))

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
        register(store, draft("diverged"))
        self.assertIsNone(
            store.offer("diverged", [90, 91, 10, 11, 99])
        )
        register(store, draft("stale"))
        self.assertIsNone(store.offer("stale", [90, 91, 10, 11, 20, 21]))

        snapshot = store.snapshot()
        self.assertEqual(snapshot.divergent_drafts, 1)
        self.assertEqual(snapshot.stale_drafts, 1)

    def test_at_most_one_thread_receives_a_proposal(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        register(store, draft("race"))
        sequence = [90, 91, 10, 11]
        with ThreadPoolExecutor(max_workers=8) as executor:
            proposals = list(
                executor.map(lambda _: store.offer("race", sequence), range(32))
            )
        self.assertEqual(sum(item is not None for item in proposals), 1)

    def test_bundle_falls_through_after_target_rejects_the_first_choice(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        store.register_bundle(
            DraftBundle(
                request_id="bundle",
                drafts=(
                    draft("bundle", tokens=(20, 21), boundary=(10,)),
                    DraftRequest(
                        request_id="bundle",
                        token_ids=(20, 99, 30),
                        boundary=DraftBoundary(token_ids=(10,)),
                        prompt_token_count=2,
                        metadata={"candidate_id": "fallback"},
                    ),
                ),
            )
        )

        first = store.offer("bundle", [90, 91, 10])
        duplicate_step = store.offer("bundle", [90, 91, 10])
        fallback = store.offer("bundle", [90, 91, 10, 20, 99])

        self.assertEqual(first.token_ids if first else (), (20, 21))
        self.assertIsNone(duplicate_step)
        self.assertEqual(fallback.token_ids if fallback else (), (30,))
        assert fallback is not None
        self.assertEqual(fallback.candidate_index, 1)
        self.assertEqual(fallback.candidate_id, "fallback")
        snapshot = store.snapshot()
        self.assertEqual(snapshot.registered_candidates, 2)
        self.assertEqual(snapshot.fallback_injections, 1)
        outcome = store.outcome("bundle")
        assert outcome is not None
        self.assertEqual(outcome.proposed_tokens, 2)
        self.assertEqual(outcome.accepted_tokens, 1)
        self.assertEqual(outcome.rejected_tokens, 1)
        self.assertEqual(outcome.unresolved_proposals, 1)
        self.assertEqual(outcome.unresolved_draft_tokens, 1)

    def test_records_direct_engine_acceptance_with_candidate_identity(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        register(
            store,
            DraftRequest(
                request_id="observed",
                token_ids=(20, 21, 22),
                boundary=DraftBoundary(token_ids=(10,)),
                prompt_token_count=2,
                metadata={
                    "candidate_id": "drafter-a",
                    "candidate_ids": ("drafter-a", "pattern-a"),
                    "sources": ("drafter", "pattern-aware"),
                },
            )
        )
        proposal = store.offer("observed", [90, 91, 10])

        self.assertEqual(proposal.token_ids if proposal else (), (20, 21, 22))
        self.assertTrue(store.observe_acceptance("observed", 2))
        self.assertFalse(store.observe_acceptance("observed", 0))
        outcome = store.take_outcome("observed")
        assert outcome is not None
        self.assertEqual(outcome.to_mapping(), {
            "num_spec_steps": 1,
            "num_draft_tokens": 3,
            "num_accepted_draft_tokens": 2,
            "num_rejected_draft_tokens": 1,
            "draft_acceptance_rate": 2 / 3,
            "mean_acceptance_length": 3.0,
            "per_step_drafted": [3],
            "per_step_accepted": [2],
            "steps": [{
                "candidate_index": 0,
                "candidate_id": "drafter-a",
                "candidate_ids": ["drafter-a", "pattern-a"],
                "sources": ["drafter", "pattern-aware"],
                "drafted_tokens": 3,
                "accepted_tokens": 2,
                "rejected_tokens": 1,
            }],
            "unresolved_proposals": 0,
            "unresolved_draft_tokens": 0,
        })
        snapshot = store.snapshot()
        self.assertEqual(snapshot.resolved_proposals, 1)
        self.assertEqual(snapshot.accepted_draft_tokens, 2)
        self.assertEqual(snapshot.rejected_draft_tokens, 1)

    def test_bundle_replacement_preserves_already_offered_candidates(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=8)
        first = DraftRequest(
            request_id="replace",
            token_ids=(20, 21),
            boundary=DraftBoundary(token_ids=(10,)),
            prompt_token_count=0,
            metadata={"candidate_id": "first"},
        )
        second = DraftRequest(
            request_id="replace",
            token_ids=(20, 99, 30),
            boundary=DraftBoundary(token_ids=(10,)),
            prompt_token_count=0,
            metadata={"candidate_id": "second"},
        )
        store.register_bundle(DraftBundle("replace", (first, second)))
        self.assertEqual(store.offer("replace", [10]).candidate_id, "first")

        receipt = store.register_bundle(
            DraftBundle("replace", (first, second, second))
        )
        self.assertIsNone(store.offer("replace", [10]))
        proposal = store.offer("replace", [10, 20, 99])

        self.assertEqual(receipt.details["candidate_count"], 2)
        self.assertEqual(receipt.details["input_candidate_count"], 3)
        self.assertEqual(proposal.candidate_id if proposal else None, "second")

    def test_tokenizes_text_boundaries_and_validates_inputs(self) -> None:
        store = BoundaryDraftStore(boundary_tokenizer=lambda text: [7, 8])
        register(
            store,
            DraftRequest(
                request_id="text",
                token_ids=(1,),
                boundary=DraftBoundary(text="boundary"),
                prompt_token_count=0,
            )
        )
        self.assertEqual(store.offer("text", [7, 8]).token_ids, (1,))

        with self.assertRaisesRegex(ValueError, "requires token_ids"):
            register(
                store,
                DraftRequest(
                    request_id="raw",
                    text="body",
                    boundary=DraftBoundary(token_ids=(1,)),
                )
            )
        with self.assertRaisesRegex(ValueError, "boundary"):
            register(
                BoundaryDraftStore(),
                DraftRequest(request_id="no-boundary", token_ids=(1,))
            )
        with self.assertRaisesRegex(ValueError, "prompt_token_count"):
            register(
                BoundaryDraftStore(),
                DraftRequest(
                    request_id="no-prompt-length",
                    token_ids=(1,),
                    boundary=DraftBoundary(token_ids=(2,)),
                )
            )
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            store.offer("text", [1], sequence_length=2)


class BoundaryDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_store_as_async_feedback(self) -> None:
        store = BoundaryDraftStore()
        feedback = BoundaryDraftFeedback(store)
        receipt = await feedback.submit(DraftBundle("async", (draft("async"),)))
        self.assertTrue(receipt.registered)
        outcome = await feedback.clear("async")
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.unresolved_proposals if outcome else None, 0)
        self.assertEqual(store.snapshot().active_requests, 0)

    async def test_submits_an_ordered_bundle(self) -> None:
        store = BoundaryDraftStore()
        feedback = BoundaryDraftFeedback(store)
        receipt = await feedback.submit(
            DraftBundle("bundle", (draft("bundle"), draft("bundle", tokens=(30,))))
        )
        self.assertEqual(receipt.details["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()

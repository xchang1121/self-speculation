from __future__ import annotations

import unittest

from examples.d3_agreement_cap_ablation import (
    _shared_prefix_length,
    analyze_agreement_cap,
    simulate_agreement_cap,
)
from examples.d3_tape_ablation import DraftOpportunity
from examples.transformers_agreement_cap_ablation import AgreementReplayStore
from examples.transformers_agreement_cap_tape_shapes import remap_candidate_shapes


class D3AgreementCapAblationTest(unittest.TestCase):
    def test_finds_the_prefix_shared_by_every_compatible_candidate(self) -> None:
        self.assertEqual(_shared_prefix_length(((1, 2, 3), (1, 2, 4), (1, 2))), 2)
        self.assertEqual(_shared_prefix_length(((1,), (2,))), 0)
        self.assertEqual(_shared_prefix_length(()), 0)

    def test_clipping_saves_wrong_tail_work_without_adding_a_target_step(self) -> None:
        treatment = simulate_agreement_cap(
            (1, 2, 3, 4),
            ((1, 2, 9, 9), (1, 2, 3, 4)),
            4,
        )

        self.assertEqual(treatment.target_steps, 2)
        self.assertEqual(treatment.proposals, 2)
        self.assertEqual(treatment.proposed_tokens, 3)
        self.assertEqual(treatment.accepted_tokens, 3)
        self.assertEqual(treatment.clipped_proposals, 1)
        self.assertEqual(treatment.clipped_tokens, 2)

    def test_clipping_an_exact_first_branch_adds_one_target_step(self) -> None:
        treatment = simulate_agreement_cap(
            (1, 2, 3, 4),
            ((1, 2, 3, 4), (1, 2, 9, 9)),
            4,
        )

        self.assertEqual(treatment.target_steps, 2)
        self.assertEqual(treatment.proposed_tokens, 3)
        self.assertEqual(treatment.accepted_tokens, 3)
        self.assertEqual(treatment.clipped_proposals, 1)

    def test_zero_agreement_falls_back_to_the_first_full_candidate(self) -> None:
        treatment = simulate_agreement_cap(
            (1, 2, 3),
            ((1, 2, 3), (9, 8, 7)),
            3,
        )

        self.assertEqual(treatment.target_steps, 1)
        self.assertEqual(treatment.proposed_tokens, 3)
        self.assertEqual(treatment.accepted_tokens, 3)
        self.assertEqual(treatment.clipped_proposals, 0)

    def test_aggregates_the_same_opportunities_under_both_policies(self) -> None:
        opportunities = (
            DraftOpportunity(
                sequence=0,
                actual_tokens=(1, 2, 3, 4),
                candidate_tokens=((1, 2, 9, 9), (1, 2, 3, 4)),
            ),
        )

        baseline = analyze_agreement_cap(opportunities, policy="full", max_draft_tokens=4)
        treatment = analyze_agreement_cap(
            opportunities,
            policy="agreement-cap",
            max_draft_tokens=4,
        )

        self.assertEqual(baseline.target_steps, treatment.target_steps)
        self.assertEqual(baseline.proposed_tokens, 5)
        self.assertEqual(treatment.proposed_tokens, 3)
        self.assertEqual(treatment.rejected_tokens, 0)

    def test_validates_policy_and_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            analyze_agreement_cap((), policy="future")
        with self.assertRaisesRegex(ValueError, "positive"):
            simulate_agreement_cap((1,), ((1,),), 0)

    def test_real_model_replay_store_clips_then_follows_the_selected_branch(self) -> None:
        store = AgreementReplayStore(
            candidates=((1, 2, 3, 4), (1, 2, 9, 9)),
            prompt_token_count=2,
            boundary_token_id=8,
            policy="agreement-cap",
            max_draft_tokens=4,
        )

        first = store.offer("request", (6, 7, 8))
        self.assertEqual(first.token_ids if first else (), (1, 2))
        self.assertTrue(store.observe_acceptance("request", 2))
        second = store.offer("request", (6, 7, 8, 1, 2, 3))
        self.assertEqual(second.token_ids if second else (), (4,))
        self.assertEqual(store.clipped_proposals, 1)
        self.assertEqual(store.clipped_tokens, 2)

    def test_tape_shape_mapping_preserves_actual_and_candidate_agreement(self) -> None:
        opportunity = DraftOpportunity(
            sequence=0,
            actual_tokens=(10, 11, 12, 13),
            candidate_tokens=((10, 11, 90, 91), (10, 11, 12, 13)),
        )

        mapped = remap_candidate_shapes(
            opportunity,
            (1, 2, 3, 4),
            vocabulary_size=32,
            limit=4,
        )

        self.assertEqual(mapped[1], (1, 2, 3, 4))
        self.assertEqual(mapped[0][:2], (1, 2))
        self.assertNotEqual(mapped[0][2], 3)


if __name__ == "__main__":
    unittest.main()

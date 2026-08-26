from __future__ import annotations

import unittest

from examples.d3_actor_history_ablation import (
    HistoryOpportunity,
    RankedDraft,
    analyze_history_caps,
    build_history_opportunities,
    ranked_policy_candidates,
    simulate_serial_drafts,
)
from examples.d3_tape_ablation import DraftOpportunity


class ActorHistoryAblationTest(unittest.TestCase):
    def test_delays_history_until_the_generated_prefix_disambiguates_it(self) -> None:
        actual = (1, 2, 3, 9, 10, 11)
        result = simulate_serial_drafts(
            actual,
            (
                RankedDraft((1, 2, 8, 8), "actor-history", 3),
                RankedDraft(actual, "actor-history", 3),
            ),
            limit=8,
        )

        self.assertEqual(result.target_steps, 4)
        self.assertEqual(result.history_proposals, 1)
        self.assertEqual(result.history_proposed_tokens, 3)
        self.assertEqual(result.history_accepted_tokens, 3)
        self.assertEqual(result.history_rejected_tokens, 0)

    def test_keeps_semantic_candidates_first_and_applies_retention_before_dedup(self) -> None:
        opportunity = HistoryOpportunity(
            tape="tape",
            sequence=2,
            actual_tokens=(1, 2),
            semantic_candidates=((7,), (8,)),
            historical_candidates=((7,), (9,), (10,)),
        )

        ranked = ranked_policy_candidates(
            opportunity,
            history_cap=2,
            min_history_candidates=1,
            min_generated_tokens=5,
            max_candidates=3,
            limit=8,
        )

        self.assertEqual([item.tokens for item in ranked], [(7,), (8,), (9,)])
        self.assertEqual([item.source for item in ranked], ["semantic", "semantic", "actor-history"])
        self.assertEqual(ranked[-1].min_generated_tokens, 5)

    def test_withholds_an_incomplete_history_group(self) -> None:
        opportunity = HistoryOpportunity(
            tape="tape",
            sequence=2,
            actual_tokens=(1, 2),
            semantic_candidates=((7,),),
            historical_candidates=((8,), (9,)),
        )

        ranked = ranked_policy_candidates(
            opportunity,
            history_cap=3,
            min_history_candidates=3,
            min_generated_tokens=1,
            max_candidates=8,
            limit=8,
        )

        self.assertEqual([item.source for item in ranked], ["semantic"])

    def test_uses_only_prior_distinct_actions_and_moves_repeats_to_the_front(self) -> None:
        opportunities = tuple(
            DraftOpportunity(sequence, actual, ())
            for sequence, actual in enumerate(((1,), (2,), (1,), (3,)))
        )

        result = build_history_opportunities("tape", opportunities)

        self.assertEqual(result[0].historical_candidates, ())
        self.assertEqual(result[1].historical_candidates, ((1,),))
        self.assertEqual(result[2].historical_candidates, ((2,), (1,)))
        self.assertEqual(result[3].historical_candidates, ((1,), (2,)))

    def test_known_exact_is_an_explicit_primary_not_a_history_leak(self) -> None:
        opportunities = (
            DraftOpportunity(1, (1, 2, 3), ((9,),)),
            DraftOpportunity(2, (4, 5, 6), ((8,),)),
        )

        result = build_history_opportunities(
            "tape",
            opportunities,
            known_primary_exact_sequences=(2,),
        )

        self.assertEqual(result[0].semantic_candidates, ((9,),))
        self.assertEqual(result[1].semantic_candidates, ((4, 5, 6), (8,)))
        self.assertEqual(result[1].historical_candidates, ((1, 2, 3),))

    def test_cap_three_recovers_a_cycle_without_regressing_other_turns(self) -> None:
        # A one-token semantic prefix provides the same causal anchor as an
        # imperfect Drafter candidate.  The next target token disambiguates the
        # three-action history before the exact fallback is offered.
        actions = (
            (1, 11, 12, 13),
            (1, 21, 22, 23),
            (1, 31, 32, 33),
            (1, 11, 12, 13),
        )
        opportunities = tuple(
            HistoryOpportunity(
                tape="cycle",
                sequence=index,
                actual_tokens=action,
                semantic_candidates=((1, 99),),
                historical_candidates=tuple(reversed(actions[:index])),
            )
            for index, action in enumerate(actions)
        )

        cap_two, cap_three = analyze_history_caps(
            opportunities,
            history_caps=(2, 3),
            min_history_candidates=3,
            min_generated_tokens=1,
            max_candidates=8,
            limit=8,
        )

        self.assertEqual(cap_two.incremental_target_steps_saved, 0)
        self.assertEqual(cap_three.incremental_target_steps_saved, 1)
        self.assertEqual(cap_three.history_accepted_tokens, 2)
        self.assertEqual(cap_three.history_rejected_tokens, 0)
        self.assertEqual(cap_three.per_opportunity_regressions, 0)


if __name__ == "__main__":
    unittest.main()

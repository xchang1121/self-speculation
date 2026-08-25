from __future__ import annotations

import unittest

from examples.d3_dynamic_cap_ablation import (
    aggregate_results,
    simulate_dynamic_cap,
)
from examples.d3_tape_ablation import DraftOpportunity


class D3DynamicCapAblationTest(unittest.TestCase):
    def test_hf_heuristic_uses_only_prior_completed_requests(self) -> None:
        opportunities = (
            DraftOpportunity(2, (1, 2, 3, 4), ((1, 2, 3, 4),)),
            DraftOpportunity(1, (1, 9, 3, 4), ((1, 2, 3, 4),)),
            DraftOpportunity(3, (1, 2, 3, 4, 5, 6), ((1, 2, 3, 4, 5, 6),)),
        )

        result = simulate_dynamic_cap(
            opportunities,
            policy="hf-heuristic",
            initial_cap=4,
            min_cap=2,
            max_cap=8,
        )

        self.assertEqual(result.cap_trace, (4, 3, 5))
        self.assertEqual(result.final_cap, 7)
        self.assertEqual(result.proposed_tokens, 12)
        self.assertEqual(result.accepted_tokens, 9)
        self.assertEqual(result.rejected_tokens, 3)

    def test_no_offer_does_not_change_the_cap(self) -> None:
        result = simulate_dynamic_cap(
            (DraftOpportunity(1, (1, 2), ()),),
            policy="hf-heuristic",
            initial_cap=6,
            min_cap=2,
            max_cap=8,
        )

        self.assertEqual(result.cap_trace, (6,))
        self.assertEqual(result.final_cap, 6)

    def test_aggregates_cold_tapes_without_sharing_future_feedback(self) -> None:
        opportunity = (DraftOpportunity(1, (1, 2, 3), ((1, 2, 3),)),)
        first = simulate_dynamic_cap(
            opportunity,
            policy="fixed",
            initial_cap=3,
            min_cap=1,
            max_cap=3,
        )
        second = simulate_dynamic_cap(
            opportunity,
            policy="fixed",
            initial_cap=3,
            min_cap=1,
            max_cap=3,
        )

        pooled = aggregate_results((first, second))

        self.assertEqual(pooled["tapes"], 2)
        self.assertEqual(pooled["accepted_tokens"], 6)
        self.assertEqual(pooled["rejected_tokens"], 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from examples.d3_tape_ablation import _order_candidates, _simulate


class D3TapeAblationTest(unittest.TestCase):
    def test_prefix_consensus_selects_the_tied_candidate_medoid(self) -> None:
        candidates = ((1, 9), (1, 2, 8, 8), (1, 2, 3), (1, 2, 4))

        self.assertEqual(
            _order_candidates(
                candidates,
                limit=3,
                ordering="prefix-consensus",
            ),
            ((1, 2, 8, 8), (1, 2, 3), (1, 2, 4), (1, 9)),
        )

    def test_prefix_consensus_is_source_neutral_and_stable_on_ties(self) -> None:
        candidates = ((1, 9), (1, 2, 3), (1, 2, 4))

        self.assertEqual(
            _order_candidates(
                candidates,
                limit=3,
                ordering="prefix-consensus",
            ),
            ((1, 2, 3), (1, 2, 4), (1, 9)),
        )

    def test_exact_truncated_draft_saves_one_step_per_accepted_token(self) -> None:
        self.assertEqual(
            _simulate((1, 2, 3, 4, 5), ((1, 2, 3, 4, 5),), 3),
            (2, 1, 3, 3),
        )

    def test_fallback_candidate_can_continue_after_a_rejected_tail(self) -> None:
        self.assertEqual(
            _simulate(
                (1, 2, 3, 4),
                ((1, 9, 9), (1, 2, 3, 4)),
                4,
            ),
            (2, 2, 5, 3),
        )


if __name__ == "__main__":
    unittest.main()

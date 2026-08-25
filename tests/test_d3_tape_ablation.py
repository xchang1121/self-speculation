from __future__ import annotations

import unittest

from examples.d3_tape_ablation import _simulate


class D3TapeAblationTest(unittest.TestCase):
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

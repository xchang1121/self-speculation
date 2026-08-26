from __future__ import annotations

import unittest
from dataclasses import asdict

from examples.d3_drafter_race_ablation import aggregate_results
from examples.d3_tape_ablation import DraftLengthResult


class D3DrafterRaceAblationTest(unittest.TestCase):
    def test_aggregate_recomputes_weighted_acceptance_and_step_reduction(self) -> None:
        first = result(actor=10, proposed=8, accepted=4, steps=6, candidates=2)
        second = result(actor=30, proposed=2, accepted=2, steps=20, candidates=1)

        self.assertEqual(
            aggregate_results((first, second), policy="first_valid"),
            {
                "policy": "first_valid",
                "tapes": 2,
                "ordering": "completion",
                "max_draft_tokens": 28,
                "opportunities": 2,
                "candidate_count": 3,
                "actor_tokens": 40,
                "proposals": 2,
                "proposed_tokens": 10,
                "accepted_tokens": 6,
                "rejected_tokens": 4,
                "acceptance_rate": 0.6,
                "target_steps": 26,
                "target_steps_saved": 14,
                "target_step_reduction": 0.35,
            },
        )

    def test_aggregate_rejects_mixed_caps(self) -> None:
        first = result(actor=10, proposed=2, accepted=1, steps=8, candidates=1)
        second = DraftLengthResult(**{**asdict(first), "max_draft_tokens": 12})

        with self.assertRaisesRegex(ValueError, "share one configuration"):
            aggregate_results((first, second), policy="first_valid")


def result(
    *, actor: int, proposed: int, accepted: int, steps: int, candidates: int
) -> DraftLengthResult:
    saved = actor - steps
    return DraftLengthResult(
        ordering="completion",
        max_draft_tokens=28,
        opportunities=1,
        candidate_count=candidates,
        actor_tokens=actor,
        proposals=1,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        rejected_tokens=proposed - accepted,
        acceptance_rate=accepted / proposed,
        target_steps=steps,
        target_steps_saved=saved,
        target_step_reduction=saved / actor,
    )


if __name__ == "__main__":
    unittest.main()

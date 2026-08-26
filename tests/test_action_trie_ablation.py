from __future__ import annotations

import unittest

from examples.action_trie_ablation import (
    Occurrence,
    analyze,
    qwen_body,
    replay_action,
    trie_proposal,
)


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        self.assert_false(add_special_tokens)
        return tuple(text.encode("utf-8"))

    @staticmethod
    def assert_false(value: bool) -> None:
        if value:
            raise AssertionError("special tokens must be disabled")


class ActionTrieAblationTest(unittest.TestCase):
    def test_frequency_then_recency_and_prefix_budget(self) -> None:
        history = (
            Occurrence((1, 2, 3, 7), 0),
            Occurrence((1, 2, 3, 8), 1),
            Occurrence((1, 2, 4, 9), 2),
        )
        self.assertEqual(trie_proposal(history, ()), ())
        self.assertEqual(trie_proposal(history, (1,)), (2,))
        self.assertEqual(trie_proposal(history, (1, 2)), (3, 8))

    def test_counts_ended_occurrences_in_probability_denominator(self) -> None:
        history = tuple(
            [Occurrence((1,), index) for index in range(10)]
            + [Occurrence((1, 2), 10)]
        )
        self.assertEqual(trie_proposal(history, (1,)), ())

    def test_requeries_after_verifier_bonus_and_preserves_target(self) -> None:
        result = replay_action((1, 2, 3, 9), (Occurrence((1, 2, 3, 4), 0),))
        self.assertEqual(result.target_steps, 3)
        self.assertEqual((result.proposals, result.proposed, result.accepted), (2, 2, 1))

    def test_analyzer_updates_history_only_after_each_action(self) -> None:
        call = {"name": "read", "arguments": {"path": "x"}}
        tokens = tuple(qwen_body(call).encode("utf-8"))
        recording = {
            "cases": [
                {
                    "case_id": "case",
                    "turns": [
                        {"turn_index": index, "call": call, "target_body_tokens": tokens, "enabled_tool_call": True, "main": {"truncated": False}}
                        for index in range(2)
                    ],
                }
            ]
        }
        result = analyze(recording, ByteTokenizer())
        self.assertEqual(result["actions"], 2)
        self.assertEqual(result["control_target_steps"], len(tokens) * 2)
        self.assertGreater(result["target_steps_saved"], 0)
        self.assertEqual(result["per_case_regressions"], 0)


if __name__ == "__main__":
    unittest.main()

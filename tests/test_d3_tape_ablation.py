from __future__ import annotations

import unittest

from examples.d3_tape_ablation import (
    ParsedExchange,
    _order_candidates,
    _simulate,
    build_opportunities,
)
from self_speculation import ToolCall, format_tool_call_draft


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

    def test_drafter_width_selects_dispatch_order_before_completion_order(self) -> None:
        tokenizer = CharacterTokenizer()
        actor_call = ToolCall(name="actual", arguments={}, format="tagged_json")
        first = ToolCall(name="first", arguments={}, format="tagged_json")
        second = ToolCall(name="second", arguments={}, format="tagged_json")
        excluded_fastest = ToolCall(
            name="excluded",
            arguments={},
            format="tagged_json",
        )
        opportunities = build_opportunities(
            (
                ParsedExchange(3, "draft", "same", 1.0, (excluded_fastest,)),
                ParsedExchange(0, "actor", "same", 100.0, (actor_call,)),
                ParsedExchange(2, "draft", "same", 20.0, (second,)),
                ParsedExchange(1, "draft", "same", 50.0, (first,)),
            ),
            actor_model="actor",
            drafter_model="draft",
            tokenizer=tokenizer,
            drafter_width=2,
        )

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(
            opportunities[0].candidate_tokens,
            (
                tokenizer.tokens(second),
                tokenizer.tokens(first),
            ),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            build_opportunities(
                (),
                actor_model="actor",
                drafter_model="draft",
                tokenizer=tokenizer,
                drafter_width=0,
            )

    def test_completion_limit_keeps_the_first_valid_dispatch_selected_response(self) -> None:
        tokenizer = CharacterTokenizer()
        actor_call = ToolCall(name="winner", arguments={}, format="tagged_json")
        slow = ToolCall(name="slow", arguments={}, format="tagged_json")
        winner = ToolCall(name="winner", arguments={}, format="tagged_json")
        excluded = ToolCall(name="excluded", arguments={}, format="tagged_json")

        opportunities = build_opportunities(
            (
                ParsedExchange(0, "actor", "same", 100.0, (actor_call,)),
                ParsedExchange(1, "draft", "same", 5.0, ()),
                ParsedExchange(2, "draft", "same", 50.0, (slow,)),
                ParsedExchange(3, "draft", "same", 20.0, (winner,)),
                ParsedExchange(4, "draft", "same", 1.0, (excluded,)),
            ),
            actor_model="actor",
            drafter_model="draft",
            tokenizer=tokenizer,
            drafter_width=3,
            drafter_completion_limit=1,
        )

        self.assertEqual(
            opportunities[0].candidate_tokens,
            (tokenizer.tokens(winner),),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            build_opportunities(
                (),
                actor_model="actor",
                drafter_model="draft",
                tokenizer=tokenizer,
                drafter_completion_limit=0,
            )


class CharacterTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        self.assert_no_special_tokens(add_special_tokens)
        return tuple(ord(character) for character in value)

    def tokens(self, call: ToolCall) -> tuple[int, ...]:
        return self.encode(
            format_tool_call_draft((call,)),
            add_special_tokens=False,
        )

    @staticmethod
    def assert_no_special_tokens(add_special_tokens: bool) -> None:
        if add_special_tokens:
            raise AssertionError("the tape analyzer must not add special tokens")


if __name__ == "__main__":
    unittest.main()

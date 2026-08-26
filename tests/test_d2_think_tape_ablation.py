from __future__ import annotations

import math
import unittest

from examples.d2_think_tape_ablation import (
    analyze_d3_recording,
    analyze_recording,
    exact_call_match,
    next_probe_index,
    normalize_text_messages,
    parse_main_tool_call,
    parse_probe_tool_call,
    span_min_probability,
)


def _call(name: str, value: str) -> dict:
    return {"name": name, "arguments": {"path": value}}


def _probe(
    call: dict | None,
    confidence: float,
    *,
    tokens: int = 10,
    runway: float = 100.0,
) -> dict:
    return {
        "call": call,
        "confidence": confidence,
        "token_count": tokens,
        "optimistic_runway_ms": runway,
    }


def _turn(main: dict, probes: list[dict]) -> dict:
    return {
        "main": {"call": main},
        "reference_calls": [main],
        "probes": probes,
    }


class D2ThinkTapeAblationTest(unittest.TestCase):
    def test_normalizes_openai_text_blocks_without_dropping_tool_fields(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "input_text", "text": "second"},
                ],
                "tool_call_id": "call-1",
            }
        ]

        self.assertEqual(
            normalize_text_messages(messages),
            [
                {
                    "role": "user",
                    "content": "first\nsecond",
                    "tool_call_id": "call-1",
                }
            ],
        )

    def test_parses_balanced_main_and_forced_prefix_probe_calls(self) -> None:
        expected = _call("read", "a.py")
        main = (
            "<think>inspect</think>\n<tool_call>\n"
            '{"name":"read","arguments":{"path":"a.py"}}\n</tool_call>'
        )
        probe = 'read", "arguments": {"path": "a.py"}} trailing'

        self.assertEqual(parse_main_tool_call(main), expected)
        self.assertEqual(parse_probe_tool_call(probe), expected)
        self.assertTrue(exact_call_match(expected, parse_probe_tool_call(probe)))

    def test_confidence_uses_only_positions_two_through_twenty_one(self) -> None:
        values = [-20.0] + [-0.1] * 19 + [-0.2, -30.0]

        self.assertAlmostEqual(span_min_probability(values), math.exp(-0.2))
        self.assertIsNone(span_min_probability([None]))

    def test_retry_schedule_waits_for_probe_and_sentence_boundary(self) -> None:
        prefixes = ["x" * index for index in range(1, 121)]
        prefixes[59] += ". "
        times = [float(index * 10) for index in range(1, 121)]

        self.assertEqual(
            next_probe_index(
                prefixes,
                times,
                last_probe_index=1,
                prior_probe_end_ms=400.0,
            ),
            60,
        )
        self.assertEqual(
            next_probe_index(
                prefixes,
                times,
                last_probe_index=1,
                prior_probe_end_ms=900.0,
            ),
            90,
        )

    def test_d2_recovers_low_confidence_d1_miss_without_oracle_substitution(self) -> None:
        actual = _call("read", "right.py")
        wrong = _call("read", "wrong.py")
        turns = [
            _turn(
                actual,
                [
                    _probe(wrong, 0.4, runway=200.0),
                    _probe(actual, 0.95, tokens=5, runway=80.0),
                ],
            )
            for _ in range(6)
        ]
        result = analyze_recording({"turns": turns})

        self.assertEqual(result["d1"]["exact_hits"], 0)
        self.assertEqual(result["d1_d2"]["exact_hits"], 6)
        self.assertEqual(result["d1_d2"]["recovered_hits"], 6)
        self.assertTrue(result["product_gate_passed"])

        high_confidence_wrong = {
            "turns": [
                _turn(
                    actual,
                    [
                        _probe(wrong, 0.95),
                        _probe(actual, 0.99),
                    ],
                )
            ]
        }
        committed = analyze_recording(high_confidence_wrong)
        self.assertEqual(committed["d1_d2"]["exact_hits"], 0)
        self.assertEqual(committed["d2_oracle_exact_hits"], 1)

    def test_delayed_policy_keeps_an_independent_token_one_d1_control(self) -> None:
        actual = _call("read", "right.py")
        turn = _turn(actual, [_probe(actual, 0.95, tokens=5)])
        turn["d1_probe"] = _probe(_call("read", "wrong.py"), 0.2, tokens=10)
        result = analyze_recording(
            {
                "config": {"phase1_span_tokens": 20},
                "turns": [turn] * 6,
            }
        )

        self.assertEqual(result["d1"]["exact_hits"], 0)
        self.assertEqual(result["d1"]["probe_tokens"], 60)
        self.assertEqual(result["d1_d2"]["exact_hits"], 6)
        self.assertEqual(
            result["d1_d2"]["efficiency_accounting"]["policy"],
            "phase1_20_token_early_abort",
        )

    def test_d3_uses_only_parseable_drafts_available_before_the_boundary(self) -> None:
        actual = _call("read", "right.py")
        turn = _turn(
            actual,
            [
                {
                    **_probe(_call("read", "wrong.py"), 0.4, tokens=2),
                    "token_ids": [1, 9],
                    "snapshot_ms": 1.0,
                    "wall_ms": 1.0,
                },
                {
                    **_probe(actual, 0.95, tokens=3),
                    "token_ids": [1, 2, 3],
                    "snapshot_ms": 2.0,
                    "wall_ms": 1.0,
                },
                {
                    **_probe(actual, 0.99, tokens=3),
                    "token_ids": [1, 2, 3],
                    "snapshot_ms": 11.0,
                    "wall_ms": 1.0,
                },
            ],
        )
        turn["main"].update(
            {
                "token_ids": [7, 8, 1, 2, 3],
                "token_times_ms": [10.0] * 5,
            }
        )
        result = analyze_d3_recording(
            {"turns": [turn]},
            MarkerTokenizer(),
        )

        self.assertEqual(result["d1"]["accepted_target_tokens"], 1)
        self.assertEqual(result["d1_d2"]["accepted_target_tokens"], 3)
        self.assertEqual(
            result["available_parseable_oracle"]["accepted_target_tokens"],
            3,
        )


class MarkerTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("marker encoding must not add special tokens")
        return [7, 8]


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from examples.d2_think_tape_ablation import (
    _estimated_probe_prefix_wall_ms,
    _request_hash,
    analyze_d3_recording,
    analyze_recording,
    exact_call_match,
    load_case_manifest,
    next_probe_index,
    normalize_text_messages,
    parse_main_tool_call,
    parse_probe_tool_call,
    record_case,
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
    def test_estimates_probe_prefix_wall_without_discarding_fixed_cost(self) -> None:
        probe = {
            "wall_ms": 120.0,
            "token_count": 100,
            "timings": {"predicted_ms": 100.0, "predicted_n": 100},
        }

        self.assertEqual(_estimated_probe_prefix_wall_ms(probe, 20), 40.0)
        self.assertEqual(
            _estimated_probe_prefix_wall_ms({**probe, "timings": {}}, 20),
            120.0,
        )

    def test_loads_integrity_checked_case_manifest(self) -> None:
        messages = [{"role": "user", "content": "inspect"}]
        tools = [{"type": "function", "function": {"name": "read"}}]
        manifest = {
            "format": "self-speculation-action-case-manifest",
            "version": 1,
            "cases": [
                {
                    "case_id": "case-1",
                    "request_hash": _request_hash(messages, tools),
                    "messages": messages,
                    "tools": tools,
                    "reference_calls": [],
                    "sources": [{"dataset": "fixture"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "cases.json")
            path.write_text(json.dumps(manifest), encoding="utf-8")
            cases, metadata = load_case_manifest(path)

            self.assertEqual(cases[0]["case_id"], "case-1")
            self.assertEqual(metadata[0]["format"], manifest["format"])

            manifest["cases"][0]["request_hash"] = "wrong"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request hash mismatch"):
                load_case_manifest(path)

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

    def test_bounded_policy_keeps_d1_full_and_uses_configured_gates(self) -> None:
        actual = _call("read", "right.py")
        exact_d1 = _turn(actual, [_probe(actual, 0.95, tokens=100)])
        recovered = _turn(
            actual,
            [
                _probe(_call("read", "wrong.py"), 0.4, tokens=100),
                _probe(actual, 0.95, tokens=10),
            ],
        )
        result = analyze_recording(
            {
                "config": {
                    "phase1_span_tokens": 20,
                    "phase1_first_probe_full": True,
                    "gate_max_mean_probe_attempts": 1.5,
                    "gate_max_probe_token_ratio": 1.25,
                },
                "turns": [exact_d1] * 3 + [recovered] * 3,
            }
        )

        self.assertEqual(result["d1"]["probe_tokens"], 600)
        self.assertEqual(result["d1_d2"]["probe_attempts"], 9)
        self.assertEqual(
            result["d1_d2"]["efficiency_accounting"]["policy"],
            "d1_full_plus_d2_phase1_20_token_early_abort",
        )
        self.assertEqual(
            result["d1_d2"]["efficiency_accounting"]["probe_tokens"],
            630,
        )
        self.assertTrue(result["product_gate_passed"])

    def test_recorder_stops_after_the_first_confident_probe(self) -> None:
        case = {
            "request_hash": "request",
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": [],
            "reference_calls": [],
            "sources": [],
        }
        result = record_case(
            FakeProbeClient([0.95, 0.95]),
            FakeProbeTokenizer(),
            case,
            main_max_tokens=100,
            probe_max_tokens=100,
            max_retries=2,
            retry_token_step=50,
            min_tokens_first_probe=1,
            stop_after_confident_probe=True,
            confidence_threshold=0.90,
            seed=42,
        )

        self.assertEqual(len(result["probes"]), 1)

        retried = record_case(
            FakeProbeClient([0.50, 0.95]),
            FakeProbeTokenizer(),
            case,
            main_max_tokens=100,
            probe_max_tokens=100,
            max_retries=2,
            retry_token_step=50,
            min_tokens_first_probe=1,
            stop_after_confident_probe=True,
            confidence_threshold=0.90,
            seed=42,
        )
        self.assertEqual(len(retried["probes"]), 2)

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

    def test_d3_bundle_gate_checks_target_steps_after_fallback(self) -> None:
        actual = _call("read", "right.py")
        turn = _turn(
            actual,
            [
                {
                    **_probe(_call("read", "wrong.py"), 0.4),
                    "token_ids": [1, 9, 9, 9],
                    "snapshot_ms": 1.0,
                    "wall_ms": 1.0,
                },
                {
                    **_probe(actual, 0.95),
                    "token_ids": [1, 2, 3, 4],
                    "snapshot_ms": 2.0,
                    "wall_ms": 1.0,
                },
            ],
        )
        turn["main"].update(
            {
                "token_ids": [7, 8, 1, 2, 3, 4],
                "token_times_ms": [10.0] * 6,
            }
        )
        result = analyze_d3_recording({"turns": [turn]}, MarkerTokenizer())
        bundle = result["bounded_d2_then_d1_bundle_k28"]

        self.assertLess(
            bundle["d1_d2"]["target_steps"],
            bundle["d1"]["target_steps"],
        )
        self.assertTrue(bundle["passed"])

        turn["probes"][0]["call"] = actual
        turn["probes"][0]["token_ids"] = [1, 2, 3, 4]
        turn["probes"][1]["call"] = _call("read", "wrong.py")
        turn["probes"][1]["token_ids"] = [1, 9, 9, 9]
        regressed = analyze_d3_recording({"turns": [turn]}, MarkerTokenizer())
        self.assertFalse(regressed["bounded_d2_then_d1_bundle_k28"]["passed"])

    def test_d3_reuses_unparseable_phase1_tokens_only_after_policy_retry(self) -> None:
        actual = _call("read", "right.py")
        turns = []
        for _ in range(6):
            turn = _turn(
                actual,
                [
                    {
                        **_probe(_call("read", "wrong.py"), 0.4, tokens=4),
                        "token_ids": [1, 9, 9, 9],
                        "snapshot_ms": 1.0,
                        "wall_ms": 1.0,
                    },
                    {
                        **_probe(None, 0.4, tokens=4),
                        "token_ids": [1, 2, 3, 4],
                        "snapshot_ms": 2.0,
                        "wall_ms": 1.0,
                    },
                ],
            )
            turn["main"].update(
                {
                    "token_ids": [7, 8, 1, 2, 3, 4],
                    "token_times_ms": [10.0] * 6,
                }
            )
            turns.append(turn)

        result = analyze_d3_recording(
            {
                "config": {
                    "phase1_span_tokens": 20,
                    "phase1_first_probe_full": True,
                },
                "turns": turns,
            },
            MarkerTokenizer(),
        )
        reuse = result["low_confidence_phase1_draft_reuse_k28"]

        self.assertEqual(reuse["eligible_policy_turns"], 6)
        self.assertEqual(reuse["available_before_boundary"], 6)
        self.assertLess(
            reuse["d1_then_phase1"]["target_steps"],
            reuse["d1"]["target_steps"],
        )
        self.assertEqual(reuse["per_turn_target_step_regressions"], 0)
        self.assertFalse(reuse["passed"])

        turns[0]["probes"][0]["confidence"] = 0.95
        no_retry = analyze_d3_recording(
            {
                "config": {
                    "phase1_span_tokens": 20,
                    "phase1_first_probe_full": True,
                },
                "turns": turns,
            },
            MarkerTokenizer(),
        )["low_confidence_phase1_draft_reuse_k28"]
        self.assertEqual(no_retry["eligible_policy_turns"], 5)


class MarkerTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("marker encoding must not add special tokens")
        return [7, 8]


class FakeProbeTokenizer:
    def apply_chat_template(self, *args, **kwargs) -> str:
        return "prompt"

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        return "x\n" * len(token_ids)


class FakeProbeClient:
    def __init__(self, confidences: list[float]) -> None:
        self.confidences = iter(confidences)

    def stream_main(self, prompt: str, *, max_tokens: int, seed: int) -> dict:
        return {
            "text": (
                '<think>inspect</think><tool_call>{"name":"read",'
                '"arguments":{"path":"right.py"}}</tool_call>'
            ),
            "token_ids": list(range(1, 101)),
            "token_times_ms": [float(index * 10) for index in range(1, 101)],
            "wall_ms": 1000.0,
            "timings": {},
        }

    def complete_probe(self, prompt: str, *, max_tokens: int, seed: int) -> dict:
        confidence = next(self.confidences)
        return {
            "text": 'read", "arguments": {"path": "right.py"}}',
            "wall_ms": 10.0,
            "token_ids": [1, 2, 3],
            "token_count": 3,
            "selected_logprobs": [0.0, math.log(confidence)],
            "timings": {},
            "tokens_cached": 0,
            "truncated": False,
        }


if __name__ == "__main__":
    unittest.main()

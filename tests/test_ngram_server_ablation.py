from __future__ import annotations

import unittest

from examples.ngram_server_ablation import (
    METRIC_NAMES,
    analyze_recording,
    parse_prometheus_metrics,
)


def _metrics(*, decode: int, drafts: int, drafted: int, accepted: int) -> dict[str, float]:
    values = {name: 0.0 for name in METRIC_NAMES}
    values["llamacpp:n_decode_total"] = float(decode)
    values["llamacpp:spec_decode_num_drafts_total"] = float(drafts)
    values["llamacpp:spec_decode_num_draft_tokens_total"] = float(drafted)
    values["llamacpp:spec_decode_num_accepted_tokens_total"] = float(accepted)
    return values


def _main(*, predicted_ms: float, wall_ms: float) -> dict:
    return {
        "token_ids": [1, 2, 3],
        "text": "same",
        "stop_type": "limit",
        "truncated": False,
        "wall_ms": wall_ms,
        "call": {"name": "read", "arguments": {"path": "a.py"}},
        "timings": {"predicted_n": 3, "predicted_ms": predicted_ms},
    }


class NgramServerAblationTest(unittest.TestCase):
    def test_parses_only_required_unlabeled_prometheus_metrics(self) -> None:
        text = "\n".join(
            [
                "# HELP ignored ignored",
                *(f"{name} {index + 0.5}" for index, name in enumerate(METRIC_NAMES)),
                'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 9',
            ]
        )

        parsed = parse_prometheus_metrics(text)

        self.assertEqual(set(parsed), set(METRIC_NAMES))
        self.assertEqual(parsed[METRIC_NAMES[0]], 0.5)

    def test_analysis_requires_lossless_output_and_all_cost_gates(self) -> None:
        pairs = [
            {
                "case_id": f"case-{index}",
                "order": ["control", "treatment"],
                "control": _main(predicted_ms=100.0, wall_ms=110.0),
                "treatment": _main(predicted_ms=90.0, wall_ms=95.0),
            }
            for index in range(2)
        ]
        recording = {
            "config": {
                "expected_pairs": 2,
                "gate_min_nonempty_pairs": 2,
                "gate_min_draft_tokens": 28,
                "gate_min_draft_acceptance": 0.5,
                "gate_max_pooled_ratio": 0.95,
                "gate_max_tail_ratio": 1.25,
            },
            "pairs": pairs,
            "metrics": {
                "delta": {
                    "control": _metrics(decode=100, drafts=0, drafted=0, accepted=0),
                    "treatment": _metrics(decode=70, drafts=10, drafted=40, accepted=30),
                }
            },
        }

        result = analyze_recording(recording)

        self.assertTrue(result["product_gate_passed"])
        self.assertEqual(result["target_forward_proxy"]["ratio"], 0.8)
        self.assertEqual(result["native_speculation"]["acceptance_rate"], 0.75)

        recording["pairs"][0]["treatment"]["token_ids"] = [9]
        mismatched = analyze_recording(recording)
        self.assertFalse(mismatched["gates"]["losslessness"])
        self.assertFalse(mismatched["product_gate_passed"])


if __name__ == "__main__":
    unittest.main()

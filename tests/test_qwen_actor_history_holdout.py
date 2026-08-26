from __future__ import annotations

import re
import unittest

from examples.qwen_actor_history_holdout import (
    BOUNDARY,
    action_key,
    analyze_recording,
    deterministic_tool_result,
    enabled_tool_call,
    qwen3_tool_body,
    target_body_tokens,
    tool_protocol_messages,
    validate_server_props,
)


class LexicalTokenizer:
    init_kwargs = {"_commit_hash": "test"}

    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        for piece in re.findall(r"[A-Za-z0-9_.]+|[^A-Za-z0-9_.\s]", text):
            if piece not in self.vocabulary:
                self.vocabulary[piece] = len(self.vocabulary) + 1
            result.append(self.vocabulary[piece])
        return result


TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
)


def call(path: str):
    return {"name": "read", "arguments": {"path": path}}


def turn(tokenizer: LexicalTokenizer, index: int, path: str):
    body = qwen3_tool_body(call(path))
    return {
        "turn_index": index,
        "call": call(path),
        "enabled_tool_call": True,
        "target_body_tokens": tokenizer.encode(body),
        "main": {"truncated": False},
    }


class QwenActorHistoryHoldoutTest(unittest.TestCase):
    def test_formats_the_frozen_spaced_qwen_body(self) -> None:
        self.assertEqual(
            qwen3_tool_body(call("a.txt")),
            '\n{"name": "read", "arguments": {"path": "a.txt"}}\n',
        )
        self.assertEqual(action_key(call("a.txt")), action_key(call("a.txt")))

    def test_mock_world_is_deterministic_and_argument_scoped(self) -> None:
        first = deterministic_tool_result(call("a.txt"))
        self.assertEqual(first, deterministic_tool_result(call("a.txt")))
        self.assertNotEqual(first, deterministic_tool_result(call("b.txt")))
        self.assertIn("a.txt", first)

    def test_adds_the_fixed_tool_policy_without_changing_the_user_task(self) -> None:
        messages = tool_protocol_messages(
            (
                {"role": "system", "content": "base"},
                {"role": "user", "content": "task"},
            )
        )

        self.assertTrue(messages[0]["content"].startswith("base"))
        self.assertIn("exactly one complete tool call", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "task"})

    def test_validates_the_frozen_server_identity(self) -> None:
        props = {
            "model_path": (
                "C:/cache/snapshots/90862c4b9d2787eaed51d12237eafdfe7c5f6077/"
                "Qwen3-1.7B-Q8_0.gguf"
            ),
            "model_ftype": "Q8_0",
            "build_info": "b10615-f280b2698",
            "total_slots": 1,
            "default_generation_settings": {
                "n_ctx": 8192,
                "params": {"speculative.types": "none"},
            },
        }

        self.assertEqual(validate_server_props(props)["context_tokens"], 8192)
        props["total_slots"] = 2
        with self.assertRaisesRegex(ValueError, "slots"):
            validate_server_props(props)

    def test_validates_enabled_tool_and_top_level_schema(self) -> None:
        self.assertTrue(enabled_tool_call(call("a.txt"), TOOLS))
        self.assertFalse(enabled_tool_call({"name": "read", "arguments": {}}, TOOLS))
        self.assertFalse(
            enabled_tool_call(
                {"name": "read", "arguments": {"path": 1}},
                TOOLS,
            )
        )
        self.assertFalse(enabled_tool_call({"name": "bash", "arguments": {}}, TOOLS))

    def test_extracts_tokens_after_the_last_boundary(self) -> None:
        prefix = [1, 2]
        marker = [ord(character) for character in BOUNDARY]
        body = [10, 11, 12]
        tokens = [*marker, 9, *prefix, *marker, *body]

        self.assertEqual(target_body_tokens(tokens, marker), tuple(body))
        self.assertIsNone(target_body_tokens([1, 2], marker))

    def test_replays_only_causal_history_and_recovers_a_fourth_action(self) -> None:
        tokenizer = LexicalTokenizer()
        recording = {
            "manifest": {"case_count": 12},
            "cases": [
                {
                    "case_id": "cycle",
                    "turns": [
                        turn(tokenizer, 0, "a.txt"),
                        turn(tokenizer, 1, "b.txt"),
                        turn(tokenizer, 2, "c.txt"),
                        turn(tokenizer, 3, "a.txt"),
                    ],
                }
            ],
        }

        result = analyze_recording(recording, tokenizer)

        self.assertEqual(result["actions"], 4)
        self.assertEqual(result["eligible_turns"], 1)
        self.assertEqual(result["exact_history_recurrences"], 1)
        self.assertEqual(result["history_proposals"], 1)
        self.assertGreater(result["history_accepted_tokens"], 0)
        self.assertEqual(result["history_rejected_tokens"], 0)
        self.assertGreater(result["target_steps_saved"], 0)

    def test_does_not_offer_a_wrong_unique_history_prefix(self) -> None:
        tokenizer = LexicalTokenizer()
        recording = {
            "manifest": {"case_count": 12},
            "cases": [
                {
                    "case_id": "miss",
                    "turns": [
                        turn(tokenizer, 0, "a.txt"),
                        turn(tokenizer, 1, "b.txt"),
                        turn(tokenizer, 2, "c.txt"),
                        turn(tokenizer, 3, "new.txt"),
                    ],
                }
            ],
        }

        result = analyze_recording(recording, tokenizer)

        self.assertEqual(result["history_proposals"], 0)
        self.assertEqual(result["target_steps_saved"], 0)
        self.assertEqual(result["per_case_regressions"], 0)


if __name__ == "__main__":
    unittest.main()

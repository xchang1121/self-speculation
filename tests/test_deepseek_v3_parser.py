from __future__ import annotations

import unittest

from self_speculation import StreamChunk, default_decoder


def call(name: str, arguments: str) -> str:
    return (
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        f"{name}\n```json\n{arguments}\n```<｜tool▁call▁end｜>"
    )


class DeepSeekV3ParserTest(unittest.TestCase):
    def test_decodes_special_token_call_across_chunks(self) -> None:
        decoder = default_decoder("deepseek_v3")
        first = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        "<｜tool▁calls▁begin｜>"
                        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n"
                        '```json\n{"q":"spo'
                    )
                )
            )
        )
        second = tuple(
            decoder.feed(
                StreamChunk(
                    text='rk"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
                )
            )
        )

        self.assertEqual(first, ())
        self.assertEqual(second[0].name, "search")
        self.assertEqual(second[0].arguments, {"q": "spork"})

    def test_reconstructs_forced_v3_function_prefix(self) -> None:
        decoder = default_decoder(
            "deepseek_v3",
            initial_text=(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                "function<｜tool▁sep｜>"
            ),
        )
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text='ping\n```json\n{}\n```<｜tool▁call▁end｜>'
                )
            )
        )
        self.assertEqual(calls[0].name, "ping")

    def test_supports_parallel_v3_calls_and_multiline_json(self) -> None:
        text = (
            "<｜tool▁calls▁begin｜>"
            + call("a", "{}")
            + call("b", '{\n  "items": [1, 2]\n}')
            + "<｜tool▁calls▁end｜>"
        )
        calls = tuple(default_decoder().feed(StreamChunk(text=text)))
        self.assertEqual([item.name for item in calls], ["a", "b"])
        self.assertEqual(calls[1].arguments, {"items": [1, 2]})

    def test_rejects_non_function_or_invalid_json(self) -> None:
        invalid_type = call("x", "{}").replace(
            "begin｜>function", "begin｜>custom"
        )
        invalid_json = call("x", "not-json")
        decoder = default_decoder("deepseek_v3")
        self.assertEqual(
            tuple(decoder.feed(StreamChunk(text=invalid_type + invalid_json))),
            (),
        )


if __name__ == "__main__":
    unittest.main()

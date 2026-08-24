from __future__ import annotations

import unittest

from self_speculation import StreamChunk, default_decoder


class DeepSeekDsmlParserTest(unittest.TestCase):
    def test_decodes_typed_dsml_parameters_across_chunks(self) -> None:
        decoder = default_decoder("deepseek_dsml")
        first = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        '<｜DSML｜tool_calls><｜DSML｜invoke name="weather">'
                        '<｜DSML｜parameter name="city" string="true">杭州'
                    )
                )
            )
        )
        second = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        '</｜DSML｜parameter>'
                        '<｜DSML｜parameter string="false" name="days">3'
                        '</｜DSML｜parameter>'
                        '<｜DSML｜parameter name="metric" string="false">true'
                        '</｜DSML｜parameter>'
                        '</｜DSML｜invoke></｜DSML｜tool_calls>'
                    )
                )
            )
        )

        self.assertEqual(first, ())
        self.assertEqual(second[0].name, "weather")
        self.assertEqual(
            second[0].arguments,
            {"city": "杭州", "days": 3, "metric": True},
        )

    def test_reconstructs_forced_dsml_invoke_prefix(self) -> None:
        decoder = default_decoder(
            "deepseek_dsml",
            initial_text='<｜DSML｜tool_calls><｜DSML｜invoke name="',
        )
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text='ping"></｜DSML｜invoke></｜DSML｜tool_calls>'
                )
            )
        )
        self.assertEqual(calls[0].name, "ping")
        self.assertEqual(calls[0].arguments, {})

    def test_supports_parallel_invokes_and_ascii_pipe_variant(self) -> None:
        calls = tuple(
            default_decoder().feed(
                StreamChunk(
                    text=(
                        '<|DSML|tool_calls>'
                        '<|DSML|invoke name="a"></|DSML|invoke>'
                        '<|DSML|invoke name="b">'
                        '<|DSML|parameter name="items" string="false">[1,2]'
                        '</|DSML|parameter></|DSML|invoke>'
                        '</|DSML|tool_calls>'
                    )
                )
            )
        )
        self.assertEqual([call.name for call in calls], ["a", "b"])
        self.assertEqual(calls[1].arguments, {"items": [1, 2]})


if __name__ == "__main__":
    unittest.main()

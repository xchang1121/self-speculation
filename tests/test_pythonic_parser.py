from __future__ import annotations

import unittest

from self_speculation import StreamChunk, default_decoder


class PythonicParserTest(unittest.TestCase):
    def test_decodes_parallel_pythonic_calls(self) -> None:
        calls = tuple(
            default_decoder("pythonic").feed(
                StreamChunk(
                    text=(
                        "<|python_tag|>["
                        "get_weather(city='Paris', units='celsius'), "
                        "get_weather(city='北京', units='celsius')]"
                    )
                )
            )
        )
        self.assertEqual([call.name for call in calls], ["get_weather", "get_weather"])
        self.assertEqual(calls[1].arguments["city"], "北京")

    def test_supports_namespaces_positionals_and_nested_literals(self) -> None:
        calls = tuple(
            default_decoder("pythonic").feed(
                StreamChunk(
                    text=(
                        "tools.search('spork', limit=3, filters={"
                        "'active': true, 'missing': null, 'scores': [-1, 2.5]})"
                    )
                )
            )
        )
        self.assertEqual(calls[0].name, "tools.search")
        self.assertEqual(calls[0].arguments["_args"], ["spork"])
        self.assertEqual(
            calls[0].arguments["filters"],
            {"active": True, "missing": None, "scores": [-1, 2.5]},
        )

    def test_waits_for_complete_streamed_expression(self) -> None:
        decoder = default_decoder("pythonic")
        self.assertEqual(
            tuple(decoder.feed(StreamChunk(text="<|python_tag|>[ping("))),
            (),
        )
        calls = tuple(decoder.feed(StreamChunk(text="), pong(x=1)]")))
        self.assertEqual([call.name for call in calls], ["ping", "pong"])

    def test_rejects_executable_or_dynamic_argument_expressions(self) -> None:
        parser = default_decoder("pythonic")
        self.assertEqual(
            tuple(parser.feed(StreamChunk(text="run(value=danger())"))),
            (),
        )
        self.assertEqual(
            tuple(
                default_decoder("pythonic").feed(
                    StreamChunk(text="run(**payload)")
                )
            ),
            (),
        )
        self.assertEqual(
            tuple(
                default_decoder("pythonic").feed(
                    StreamChunk(text="[x for x in values]")
                )
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()

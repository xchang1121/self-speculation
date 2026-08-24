from __future__ import annotations

import unittest

from self_speculation import StreamChunk, default_decoder


class QwenXmlParserTest(unittest.TestCase):
    def test_decodes_qwen_xml_across_chunks(self) -> None:
        decoder = default_decoder("qwen_xml")
        first = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        "<tool_call>\n<function=search>\n"
                        "<parameter=query>\nSPORK"
                    )
                )
            )
        )
        second = tuple(
            decoder.feed(
                StreamChunk(
                    text="\n</parameter>\n</function>\n</tool_call>"
                )
            )
        )

        self.assertEqual(first, ())
        self.assertEqual(second[0].name, "search")
        self.assertEqual(second[0].arguments, {"query": "SPORK"})

    def test_reconstructs_forced_function_prefix(self) -> None:
        decoder = default_decoder(
            "qwen_xml", initial_text="<tool_call>\n<function="
        )
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        "lookup>\n<parameter=id>\n7\n</parameter>"
                        "\n</function>\n</tool_call>"
                    )
                )
            )
        )
        self.assertEqual(calls[0].name, "lookup")
        self.assertEqual(calls[0].arguments, {"id": "7"})

    def test_supports_multiple_xml_function_calls(self) -> None:
        decoder = default_decoder()
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        "<tool_call><function=a></function></tool_call>"
                        "<tool_call><function=b>"
                        "<parameter=x>2</parameter></function></tool_call>"
                    )
                )
            )
        )
        self.assertEqual(
            [(call.index, call.name, call.arguments) for call in calls],
            [(0, "a", {}), (1, "b", {"x": "2"})],
        )


if __name__ == "__main__":
    unittest.main()

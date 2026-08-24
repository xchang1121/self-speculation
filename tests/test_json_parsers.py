from __future__ import annotations

import unittest

from self_speculation import StreamChunk, default_decoder


class TaggedJsonParserTest(unittest.TestCase):
    def test_decodes_tagged_json_across_stream_chunks(self) -> None:
        decoder = default_decoder("tagged_json")

        first = tuple(
            decoder.feed(
                StreamChunk(text='<tool_call>{"name":"search","arguments":{"q":')
            )
        )
        second = tuple(
            decoder.feed(StreamChunk(text='"spork"}}</tool_call>'))
        )

        self.assertEqual(first, ())
        self.assertEqual(second[0].name, "search")
        self.assertEqual(second[0].arguments, {"q": "spork"})
        self.assertEqual(second[0].format, "tagged_json")

    def test_reconstructs_output_generated_after_a_forced_prefix(self) -> None:
        decoder = default_decoder(
            "tagged_json", initial_text='<tool_call>\n{"name":"'
        )

        calls = tuple(
            decoder.feed(
                StreamChunk(text='lookup","arguments":{"id":7}}</tool_call>')
            )
        )

        self.assertEqual(calls[0].name, "lookup")
        self.assertEqual(calls[0].arguments, {"id": 7})

    def test_decodes_multiple_tagged_calls(self) -> None:
        decoder = default_decoder("tagged_json")
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
                        '<tool_call>{"name":"b","arguments":{}}</tool_call>'
                    )
                )
            )
        )
        self.assertEqual([(call.index, call.name) for call in calls], [(0, "a"), (1, "b")])


class JsonBranchTest(unittest.TestCase):
    def test_mistral_parallel_array(self) -> None:
        decoder = default_decoder("mistral_json")
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        '[TOOL_CALLS] ['
                        '{"name":"weather","arguments":{"city":"Paris"}},'
                        '{"name":"time","arguments":{"zone":"UTC"}}]'
                    )
                )
            )
        )
        self.assertEqual([call.name for call in calls], ["weather", "time"])

    def test_llama_openai_nested_function_shape(self) -> None:
        decoder = default_decoder("llama_json")
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text=(
                        '<|python_tag|>{"id":"c1","type":"function",'
                        '"function":{"name":"search","arguments":"{\\"q\\":\\"x\\"}"}}'
                    )
                )
            )
        )
        self.assertEqual(calls[0].call_id, "c1")
        self.assertEqual(calls[0].arguments, {"q": "x"})

    def test_xlam_code_block_and_direct_array(self) -> None:
        code_decoder = default_decoder("xlam_json")
        code_calls = tuple(
            code_decoder.feed(
                StreamChunk(
                    text='```json\n[{"name":"search","parameters":{"q":"x"}}]\n```'
                )
            )
        )
        bare_calls = tuple(
            default_decoder("bare_json").feed(
                StreamChunk(text='[{"name":"ping","arguments":{}}]')
            )
        )
        self.assertEqual(code_calls[0].arguments, {"q": "x"})
        self.assertEqual(bare_calls[0].name, "ping")

    def test_incomplete_or_unrelated_json_is_not_a_call(self) -> None:
        incomplete = default_decoder("tagged_json")
        self.assertEqual(
            tuple(incomplete.feed(StreamChunk(text='<tool_call>{"name":'))),
            (),
        )
        unrelated = default_decoder("bare_json")
        self.assertEqual(
            tuple(unrelated.feed(StreamChunk(text='{"answer":42}'))),
            (),
        )


if __name__ == "__main__":
    unittest.main()

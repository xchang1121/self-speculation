from __future__ import annotations

import unittest
from collections.abc import Iterable

from self_speculation import (
    ParserRegistrationError,
    ParserRegistry,
    StreamChunk,
    StreamingToolCallDecoder,
    ToolCall,
    ToolCallDelta,
)


class MarkerParser:
    name = "marker"

    def parse(self, text: str) -> Iterable[ToolCall]:
        if "CALL:search" in text:
            return (ToolCall("search", {"q": "x"}, format=self.name),)
        return ()


class OtherParser:
    name = "other"

    def parse(self, text: str) -> Iterable[ToolCall]:
        if "OTHER" in text:
            return (ToolCall("other", {}, format=self.name),)
        return ()


class ParserRegistryTest(unittest.TestCase):
    def test_registers_orders_and_selects_custom_parsers(self) -> None:
        registry = ParserRegistry()
        registry.register("marker", MarkerParser)
        registry.register("other", OtherParser)

        self.assertEqual(registry.names(), ("marker", "other"))
        self.assertEqual(
            [parser.name for parser in registry.create_parsers(("other",))],
            ["other"],
        )
        self.assertEqual(
            [parser.name for parser in registry.create_parsers("marker")],
            ["marker"],
        )
        with self.assertRaises(ParserRegistrationError):
            registry.register("marker", MarkerParser)
        with self.assertRaises(ParserRegistrationError):
            registry.create_parsers(("missing",))

    def test_decoder_locks_to_first_matching_branch_and_deduplicates(self) -> None:
        registry = ParserRegistry()
        registry.register("marker", MarkerParser)
        registry.register("other", OtherParser)
        decoder = registry.decoder(initial_text="prefix ")

        first = tuple(decoder.feed(StreamChunk(text="CALL:search")))
        duplicate = tuple(decoder.feed(StreamChunk(text=" OTHER")))

        self.assertEqual([call.name for call in first], ["search"])
        self.assertEqual(duplicate, ())


class StructuredDeltaDecoderTest(unittest.TestCase):
    def test_assembles_fragmented_structured_call(self) -> None:
        decoder = StreamingToolCallDecoder()

        self.assertEqual(
            tuple(
                decoder.feed(
                    StreamChunk(
                        tool_call_deltas=(
                            ToolCallDelta(
                                index=0,
                                call_id="call-1",
                                name="search",
                                arguments='{"q":',
                            ),
                        )
                    )
                )
            ),
            (),
        )
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    tool_call_deltas=(ToolCallDelta(index=0, arguments='"spork"}'),),
                    finish_reason="tool_calls",
                )
            )
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].arguments, {"q": "spork"})
        self.assertEqual(calls[0].call_id, "call-1")

    def test_preserves_parallel_calls_by_index(self) -> None:
        decoder = StreamingToolCallDecoder()
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    tool_call_deltas=(
                        ToolCallDelta(0, "a", "first", '{"x":1}'),
                        ToolCallDelta(1, "b", "second", '{"x":1}'),
                    ),
                    finish_reason="tool_calls",
                )
            )
        )

        self.assertEqual([(call.index, call.name) for call in calls], [(0, "first"), (1, "second")])

    def test_structured_deltas_take_priority_over_text_parser(self) -> None:
        decoder = StreamingToolCallDecoder((MarkerParser(),))
        calls = tuple(
            decoder.feed(
                StreamChunk(
                    text="CALL:search",
                    tool_call_deltas=(ToolCallDelta(name="actual", arguments="{}"),),
                    finish_reason="tool_calls",
                )
            )
        )
        self.assertEqual([call.name for call in calls], ["actual"])

    def test_finish_completes_no_argument_call_and_is_idempotent(self) -> None:
        decoder = StreamingToolCallDecoder()
        decoder.feed(StreamChunk(tool_call_deltas=(ToolCallDelta(name="ping"),)))
        calls = tuple(decoder.finish())

        self.assertEqual(calls[0].arguments, {})
        self.assertEqual(tuple(decoder.finish()), ())
        with self.assertRaises(RuntimeError):
            decoder.feed(StreamChunk(text="late"))


if __name__ == "__main__":
    unittest.main()

"""Factories for the built-in, dependency-free parser set."""

from __future__ import annotations

from collections.abc import Sequence

from .json_parser import (
    bare_json_parser,
    llama_json_parser,
    mistral_json_parser,
    tagged_json_parser,
    xlam_json_parser,
)
from .registry import ParserRegistry, StreamingToolCallDecoder
from .qwen_xml import QwenXmlToolCallParser


def default_parser_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register("qwen_xml", QwenXmlToolCallParser)
    registry.register("tagged_json", tagged_json_parser)
    registry.register("mistral_json", mistral_json_parser)
    registry.register("llama_json", llama_json_parser)
    registry.register("xlam_json", xlam_json_parser)
    registry.register("bare_json", bare_json_parser)
    return registry


def default_decoder(
    names: str | Sequence[str] = "auto",
    *,
    initial_text: str = "",
    max_buffer_chars: int = 1_000_000,
) -> StreamingToolCallDecoder:
    return default_parser_registry().decoder(
        names,
        initial_text=initial_text,
        max_buffer_chars=max_buffer_chars,
    )

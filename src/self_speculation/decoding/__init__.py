"""Streaming tool-call decoder contracts and parser implementations."""

from .base import DecoderFactory, ToolCallDecoder
from .defaults import default_decoder, default_parser_registry
from .json_parser import JsonToolCallParser
from .registry import (
    ParserFactory,
    ParserRegistrationError,
    ParserRegistry,
    StreamingToolCallDecoder,
    TextToolCallParser,
)

__all__ = [
    "DecoderFactory",
    "JsonToolCallParser",
    "ParserFactory",
    "ParserRegistrationError",
    "ParserRegistry",
    "StreamingToolCallDecoder",
    "TextToolCallParser",
    "ToolCallDecoder",
    "default_decoder",
    "default_parser_registry",
]

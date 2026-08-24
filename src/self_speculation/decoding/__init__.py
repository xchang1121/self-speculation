"""Streaming tool-call decoder contracts and parser implementations."""

from .base import DecoderFactory, ToolCallDecoder
from .defaults import default_decoder, default_parser_registry
from .deepseek_dsml import DeepSeekDsmlToolCallParser
from .json_parser import JsonToolCallParser
from .qwen_xml import QwenXmlToolCallParser
from .registry import (
    ParserFactory,
    ParserRegistrationError,
    ParserRegistry,
    StreamingToolCallDecoder,
    TextToolCallParser,
)

__all__ = [
    "DecoderFactory",
    "DeepSeekDsmlToolCallParser",
    "JsonToolCallParser",
    "ParserFactory",
    "ParserRegistrationError",
    "ParserRegistry",
    "QwenXmlToolCallParser",
    "StreamingToolCallDecoder",
    "TextToolCallParser",
    "ToolCallDecoder",
    "default_decoder",
    "default_parser_registry",
]

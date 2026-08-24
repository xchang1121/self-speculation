"""Streaming tool-call decoder contracts and parser implementations."""

from .base import DecoderFactory, ToolCallDecoder
from .defaults import default_decoder, default_parser_registry
from .deepseek_dsml import DeepSeekDsmlToolCallParser
from .deepseek_v3 import DeepSeekV3ToolCallParser
from .json_parser import JsonToolCallParser
from .qwen_xml import QwenXmlToolCallParser
from .pythonic import PythonicToolCallParser
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
    "DeepSeekV3ToolCallParser",
    "JsonToolCallParser",
    "ParserFactory",
    "ParserRegistrationError",
    "ParserRegistry",
    "QwenXmlToolCallParser",
    "PythonicToolCallParser",
    "StreamingToolCallDecoder",
    "TextToolCallParser",
    "ToolCallDecoder",
    "default_decoder",
    "default_parser_registry",
]

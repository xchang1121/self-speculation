"""Parser for Qwen3/Qwen3.5 XML-style tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import ToolCall


_FUNCTION_RE = re.compile(
    r"<\s*function\s*=\s*([^>]+?)\s*>(.*?)<\s*/\s*function\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]+?)\s*>(.*?)<\s*/\s*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _trim_wrapping_newline(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


@dataclass(frozen=True, slots=True)
class QwenXmlToolCallParser:
    """Decode ``<function=...><parameter=...>`` branches."""

    name: str = "qwen_xml"

    def parse(self, text: str) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index, match in enumerate(_FUNCTION_RE.finditer(text)):
            function_name = match.group(1).strip()
            if not function_name:
                continue
            body = match.group(2)
            arguments = {
                parameter.group(1).strip(): _trim_wrapping_newline(
                    parameter.group(2)
                )
                for parameter in _PARAMETER_RE.finditer(body)
                if parameter.group(1).strip()
            }
            calls.append(
                ToolCall(
                    name=function_name,
                    arguments=arguments,
                    index=index,
                    format=self.name,
                    raw=match.group(0),
                )
            )
        return tuple(calls)

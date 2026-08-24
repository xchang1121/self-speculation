"""Parser for DeepSeek V3/R1 special-token tool-call output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models import ToolCall


_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>"
    r"(?P<type>.*?)<｜tool▁sep｜>"
    r"(?P<name>.*?)\r?\n```json\r?\n"
    r"(?P<arguments>.*?)\r?\n```"
    r"<｜tool▁call▁end｜>",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class DeepSeekV3ToolCallParser:
    name: str = "deepseek_v3"

    def parse(self, text: str) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index, match in enumerate(_CALL_RE.finditer(text)):
            tool_type = match.group("type").strip()
            function_name = match.group("name").strip()
            if tool_type and tool_type != "function":
                continue
            if not function_name:
                continue
            raw_arguments = match.group("arguments").strip()
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                continue
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

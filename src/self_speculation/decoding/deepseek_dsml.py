"""Parser for DeepSeek V4's DSML tool-call branch."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall


_DSML = r"(?:\||｜)DSML(?:\||｜)"
_INVOKE_RE = re.compile(
    rf'<\s*{_DSML}\s*invoke\s+name="([^"]+)"\s*>'
    rf"(.*?)"
    rf"<\s*/\s*{_DSML}\s*invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    rf"<\s*{_DSML}\s*parameter\b([^>]*)>"
    rf"(.*?)"
    rf"<\s*/\s*{_DSML}\s*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(r'([\w.-]+)\s*=\s*"([^"]*)"')


def _typed_value(raw: str, is_string: bool) -> Any:
    if is_string:
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


@dataclass(frozen=True, slots=True)
class DeepSeekDsmlToolCallParser:
    name: str = "deepseek_dsml"

    def parse(self, text: str) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index, invoke in enumerate(_INVOKE_RE.finditer(text)):
            arguments: dict[str, Any] = {}
            for parameter in _PARAMETER_RE.finditer(invoke.group(2)):
                attributes = dict(_ATTRIBUTE_RE.findall(parameter.group(1)))
                parameter_name = attributes.get("name")
                if not parameter_name:
                    continue
                arguments[parameter_name] = _typed_value(
                    parameter.group(2),
                    attributes.get("string", "true").lower() == "true",
                )
            calls.append(
                ToolCall(
                    name=invoke.group(1).strip(),
                    arguments=arguments,
                    index=index,
                    format=self.name,
                    raw=invoke.group(0),
                )
            )
        return tuple(calls)

"""Canonical boundary-relative draft bodies for built-in parser branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall
from .base import DraftBoundary


@dataclass(frozen=True, slots=True)
class ToolCallSyntax:
    """Text framing shared by probe continuation and D3 serialization."""

    boundary: str
    close: str | None = None


_TOOL_CALL_SYNTAXES = {
    "tagged_json": ToolCallSyntax("<tool_call>", "</tool_call>"),
    "qwen_xml": ToolCallSyntax("<tool_call>", "</tool_call>"),
    "deepseek_dsml": ToolCallSyntax(
        "<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"
    ),
    "deepseek_v3": ToolCallSyntax(
        "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"
    ),
    "mistral_json": ToolCallSyntax("[TOOL_CALLS]"),
    "llama_json": ToolCallSyntax("<|python_tag|>"),
    "pythonic": ToolCallSyntax("<|python_tag|>"),
}


def tool_call_syntax(format_name: str) -> ToolCallSyntax:
    """Return the canonical framing for a registered output format."""

    normalized = format_name.strip().lower()
    if not normalized:
        raise ValueError("format_name must not be empty")
    return _TOOL_CALL_SYNTAXES.get(
        normalized,
        _TOOL_CALL_SYNTAXES["tagged_json"],
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_call(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": call.arguments}


def _qwen_function(call: ToolCall) -> str:
    body = f"<function={call.name}>"
    arguments = call.arguments if isinstance(call.arguments, dict) else {"value": call.arguments}
    for key, value in arguments.items():
        rendered = value if isinstance(value, str) else _json(value)
        body += f"\n<parameter={key}>\n{rendered}\n</parameter>"
    return body + "\n</function>"


def _dsml_invoke(call: ToolCall) -> str:
    body = f'<｜DSML｜invoke name="{call.name}">'
    arguments = call.arguments if isinstance(call.arguments, dict) else {"value": call.arguments}
    for key, value in arguments.items():
        is_string = isinstance(value, str)
        rendered = value if is_string else _json(value)
        flag = "true" if is_string else "false"
        body += (
            f'\n<｜DSML｜parameter name="{key}" string="{flag}">'
            f"{rendered}</｜DSML｜parameter>"
        )
    return body + "\n</｜DSML｜invoke>"


def _deepseek_v3_call(call: ToolCall) -> str:
    return (
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        f"{call.name}\n```json\n{_json(call.arguments)}\n```"
        "<｜tool▁call▁end｜>"
    )


def _pythonic(calls: tuple[ToolCall, ...]) -> str:
    if all(call.raw for call in calls):
        rendered = [call.raw for call in calls]
    else:
        rendered = [
            f"{call.name}(arguments={repr(call.arguments)})" for call in calls
        ]
    return rendered[0] if len(rendered) == 1 else "[" + ", ".join(rendered) + "]"


def format_tool_call_draft(tool_calls: tuple[ToolCall, ...]) -> str:
    """Serialize calls after, but not including, their tool-call boundary."""

    if not tool_calls:
        raise ValueError("tool_calls must not be empty")
    format_name = tool_calls[0].format
    if format_name == "qwen_xml":
        bodies = [_qwen_function(call) for call in tool_calls]
        return ("</tool_call>\n<tool_call>\n").join(bodies)
    if format_name == "deepseek_dsml":
        return "\n".join(_dsml_invoke(call) for call in tool_calls)
    if format_name == "deepseek_v3":
        return "\n".join(_deepseek_v3_call(call) for call in tool_calls)
    if format_name == "pythonic":
        return _pythonic(tool_calls)

    value: Any = (
        _json_call(tool_calls[0])
        if len(tool_calls) == 1
        else [_json_call(call) for call in tool_calls]
    )
    return "\n" + _json(value)


def default_draft_boundary(tool_calls: tuple[ToolCall, ...]) -> DraftBoundary:
    if not tool_calls:
        raise ValueError("tool_calls must not be empty")
    return DraftBoundary(text=tool_call_syntax(tool_calls[0].format).boundary)

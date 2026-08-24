"""Streaming-friendly parsers for the common JSON tool-call branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall


_CLOSING_MARKERS = {
    "<tool_call>": "</tool_call>",
    "<think>": "</think>",
    "```json": "```",
}


def _arguments(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        entries: list[dict[str, Any]] = []
        for item in value:
            entries.extend(_entries(item))
        return entries
    if not isinstance(value, dict):
        return []

    for container_key in ("tool_calls", "calls", "tools"):
        nested = value.get(container_key)
        if isinstance(nested, list):
            return _entries(nested)

    function = value.get("function")
    if isinstance(function, dict) and function.get("name"):
        return [
            {
                "name": function["name"],
                "arguments": function.get(
                    "arguments", function.get("parameters", function.get("input", {}))
                ),
                "call_id": value.get("id") or value.get("call_id"),
                "index": value.get("index"),
            }
        ]

    name = value.get("name") or value.get("tool_name")
    if not name:
        return []
    return [
        {
            "name": name,
            "arguments": value.get(
                "arguments", value.get("parameters", value.get("input", {}))
            ),
            "call_id": value.get("id") or value.get("call_id"),
            "index": value.get("index"),
        }
    ]


def _decode_segment(segment: str) -> tuple[Any, str] | None:
    openers = [position for position in (segment.find("{"), segment.find("[")) if position >= 0]
    if not openers:
        return None
    start = min(openers)
    try:
        value, end = json.JSONDecoder().raw_decode(segment, start)
    except json.JSONDecodeError:
        return None
    return value, segment[start:end]


@dataclass(frozen=True, slots=True)
class JsonToolCallParser:
    """Parse JSON values after one of a set of branch markers."""

    name: str
    markers: tuple[str, ...] = ()
    allow_bare: bool = False

    def _segments(self, text: str) -> list[str]:
        segments: list[str] = []
        for marker in self.markers:
            offset = 0
            while True:
                marker_at = text.find(marker, offset)
                if marker_at < 0:
                    break
                start = marker_at + len(marker)
                closing = _CLOSING_MARKERS.get(marker)
                end = text.find(closing, start) if closing else -1
                segments.append(text[start:] if end < 0 else text[start:end])
                offset = start

        if self.allow_bare:
            stripped = text.lstrip()
            if stripped.startswith(("{", "[")):
                segments.append(stripped)
        return segments

    def parse(self, text: str) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        next_index = 0
        for segment in self._segments(text):
            decoded = _decode_segment(segment)
            if decoded is None:
                continue
            value, raw = decoded
            for entry in _entries(value):
                explicit_index = entry.get("index")
                index = explicit_index if isinstance(explicit_index, int) else next_index
                calls.append(
                    ToolCall(
                        name=str(entry["name"]),
                        arguments=_arguments(entry.get("arguments")),
                        call_id=entry.get("call_id"),
                        index=index,
                        format=self.name,
                        raw=raw,
                    )
                )
                next_index = max(next_index, index + 1)
        return tuple(calls)


def tagged_json_parser() -> JsonToolCallParser:
    """Hermes, Qwen2/3 JSON and SPORK's original tagged branch."""

    return JsonToolCallParser("tagged_json", ("<tool_call>",))


def mistral_json_parser() -> JsonToolCallParser:
    return JsonToolCallParser("mistral_json", ("[TOOL_CALLS]",))


def llama_json_parser() -> JsonToolCallParser:
    return JsonToolCallParser("llama_json", ("<|python_tag|>",))


def xlam_json_parser() -> JsonToolCallParser:
    return JsonToolCallParser(
        "xlam_json",
        ("```json", "<tool_call>", "[TOOL_CALLS]", "<think>"),
        allow_bare=True,
    )


def bare_json_parser() -> JsonToolCallParser:
    return JsonToolCallParser("bare_json", allow_bare=True)

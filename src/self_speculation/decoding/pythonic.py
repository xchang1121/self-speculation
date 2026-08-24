"""Safe AST parser for Llama-style Pythonic tool calls."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall


_JSON_NAMES = {"null": None, "true": True, "false": False}


class _UnsafeExpression(ValueError):
    pass


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    raise _UnsafeExpression("tool name must be an identifier or dotted name")


def _literal(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (str, int, float, bool)):
            return node.value
        raise _UnsafeExpression("unsupported constant")
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise _UnsafeExpression("dictionary unpacking is not allowed")
        return {
            _literal(key): _literal(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _UnsafeExpression("unary operators require a number")
        return value if isinstance(node.op, ast.UAdd) else -value
    raise _UnsafeExpression(f"unsafe argument expression: {type(node).__name__}")


def _calls(root: ast.expr) -> list[ast.Call]:
    if isinstance(root, ast.Call):
        return [root]
    if isinstance(root, (ast.List, ast.Tuple)) and all(
        isinstance(item, ast.Call) for item in root.elts
    ):
        return list(root.elts)  # type: ignore[return-value]
    raise _UnsafeExpression("expected one tool call or a list of tool calls")


@dataclass(frozen=True, slots=True)
class PythonicToolCallParser:
    name: str = "pythonic"
    marker: str = "<|python_tag|>"
    allow_bare: bool = True

    def _expression(self, text: str) -> str | None:
        marker_at = text.find(self.marker)
        if marker_at >= 0:
            return text[marker_at + len(self.marker) :].strip()
        stripped = text.strip()
        if self.allow_bare and stripped.startswith(("[", "(")):
            return stripped
        if self.allow_bare and stripped[:1].isidentifier():
            return stripped
        return None

    def parse(self, text: str) -> tuple[ToolCall, ...]:
        expression = self._expression(text)
        if not expression:
            return ()
        try:
            root = ast.parse(expression, mode="eval").body
            call_nodes = _calls(root)
            calls: list[ToolCall] = []
            for index, node in enumerate(call_nodes):
                arguments: dict[str, Any] = {}
                if node.args:
                    arguments["_args"] = [_literal(argument) for argument in node.args]
                for keyword in node.keywords:
                    if keyword.arg is None:
                        raise _UnsafeExpression("keyword unpacking is not allowed")
                    arguments[keyword.arg] = _literal(keyword.value)
                calls.append(
                    ToolCall(
                        name=_name(node.func),
                        arguments=arguments,
                        index=index,
                        format=self.name,
                        raw=ast.get_source_segment(expression, node) or expression,
                    )
                )
            return tuple(calls)
        except (SyntaxError, TypeError, ValueError, _UnsafeExpression):
            return ()

"""The single content renderer shared by training, evaluation, and serving."""

from __future__ import annotations

from pydantic import JsonValue

INDENT = "  "


def render_content(value: JsonValue) -> str:
    """Render a JSON value without JSON punctuation, preserving key order."""
    return "\n".join(_render_lines(value, 0))


def _render_lines(value: JsonValue, depth: int) -> list[str]:
    pad = INDENT * depth
    if _is_scalar(value):
        return [pad + _render_scalar(value)]
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if _is_scalar(item):
                text = _render_scalar(item)
                lines.append(f"{pad}- {text}" if text else f"{pad}-")
            else:
                lines.append(f"{pad}-")
                lines.extend(_render_lines(item, depth + 1))
        return lines
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if _is_scalar(item):
                text = _render_scalar(item)
                lines.append(f"{pad}{key}: {text}" if text else f"{pad}{key}:")
            else:
                lines.append(f"{pad}{key}:")
                lines.extend(_render_lines(item, depth + 1))
        return lines
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _is_scalar(value: JsonValue) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _render_scalar(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

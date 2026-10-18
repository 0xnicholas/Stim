"""The single content renderer shared by training, evaluation, and serving.

Rendered text is part of the model input contract: it is frozen with the artifact and
must be produced by these functions everywhere, never reimplemented downstream. The
module depends only on pydantic's JSON type and the standard library.
"""

from __future__ import annotations

import re

from pydantic import JsonValue

INDENT = "  "
_BROKEN_BAR = "\u00a6"
_DELIMITER = re.compile(r"<\|([A-Za-z0-9_.-]{1,64})\|>")


def render_content(value: JsonValue) -> str:
    """Render JSON as plain text, preserving key order and rewriting delimiters."""
    return "\n".join(_render_lines(value, 0))


def render_option(key: str, description: JsonValue) -> str:
    """Render an answer option as its delimiter-safe key and optional description."""
    safe_key = _rewrite_delimiters(key)
    rendered = render_content(description)
    if not rendered:
        return safe_key
    if "\n" not in rendered:
        return f"{safe_key}: {rendered}"
    indented = "\n".join(f"{INDENT}{line}" for line in rendered.splitlines())
    return f"{safe_key}:\n{indented}"


def _render_lines(value: JsonValue, depth: int) -> list[str]:
    pad = INDENT * depth
    if _is_scalar(value):
        return [pad + _render_scalar(value)]
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            lines.extend(_render_entry(pad, "-", item, depth))
        return lines
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            lines.extend(_render_entry(pad, f"{_rewrite_delimiters(key)}:", item, depth))
        return lines
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _render_entry(pad: str, marker: str, value: JsonValue, depth: int) -> list[str]:
    if _is_scalar(value):
        text = _render_scalar(value)
        return [f"{pad}{marker} {text}" if text else f"{pad}{marker}"]
    return [f"{pad}{marker}", *_render_lines(value, depth + 1)]


def _is_scalar(value: JsonValue) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _render_scalar(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _rewrite_delimiters(value)
    return str(value)


def _rewrite_delimiters(text: str) -> str:
    """Rewrite special-token literals so caller text cannot forge a delimiter."""
    return _DELIMITER.sub(lambda match: f"<{_BROKEN_BAR}{match.group(1)}{_BROKEN_BAR}>", text)

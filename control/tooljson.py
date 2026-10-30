"""Strict parser for JSON emitted by local tools.

Python's default decoder accepts duplicate keys (last value wins) and
non-finite numbers (NaN/Infinity). Both are unsafe at a trust boundary:
a second value can silently replace evidence, and NaN bypasses ordinary
range comparisons. Tool output is therefore parsed with duplicate keys
and non-finite constants rejected before the strict Pydantic model.
"""

from __future__ import annotations

import json

from control.schemas import ToolResult

MAX_TOOL_JSON_CHARS = 1_000_000


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    payload: dict = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def parse_tool_json(raw: str) -> ToolResult:
    """Parse and validate a tool payload without permissive coercion."""
    if not isinstance(raw, str):
        raise ValueError("tool output is not text")
    if len(raw) > MAX_TOOL_JSON_CHARS:
        raise ValueError(
            f"tool JSON exceeds {MAX_TOOL_JSON_CHARS} characters"
        )
    payload = json.loads(
        raw,
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("tool output is not a JSON object")
    return ToolResult.model_validate(payload)

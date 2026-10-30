"""Adversarial tool-output tests: duplicate keys and non-finite numbers."""

import pytest

from control.tooljson import MAX_TOOL_JSON_CHARS, parse_tool_json


def test_duplicate_tool_keys_and_non_finite_numbers_are_rejected():
    duplicate = (
        '{"status": "success", "disease_name": "Melanoma", '
        '"confidence_score": 0.8, "confidence_score": 0.1}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_tool_json(duplicate)

    for constant in ("NaN", "Infinity", "-Infinity"):
        raw = (
            '{"status": "success", "disease_name": "Melanoma", '
            f'"confidence_score": {constant}}}'
        )
        with pytest.raises(ValueError, match="non-finite JSON number"):
            parse_tool_json(raw)


def test_oversized_tool_json_is_rejected_before_parsing():
    raw = "{} " + " " * MAX_TOOL_JSON_CHARS
    with pytest.raises(ValueError, match="exceeds"):
        parse_tool_json(raw)

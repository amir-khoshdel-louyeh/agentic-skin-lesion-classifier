"""Extract the first complete JSON object from a model reply.

Every role used to run `re.search(r"\\{.*\\}", raw, DOTALL)`, which spans
from the FIRST brace to the LAST one. A reply containing two objects, or
prose with a stray brace after the object, then fails to parse even though
a perfectly good verdict is sitting in the text — the session loses its
evidence and burns a repair retry. `raw_decode` stops at the end of the
first complete object instead, so trailing text (or a second object) is
irrelevant.
"""

from __future__ import annotations

import json
from typing import Any


def first_json_object(raw: str) -> dict[str, Any] | None:
    """First complete JSON object in `raw`, or None when there is none."""
    text = (raw or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            return None
        try:
            value, _end = decoder.raw_decode(text, start)
        except ValueError:
            # Not a valid object starting here (e.g. "{}" in prose, or a
            # truncated blob): try the next brace instead of giving up.
            index = start + 1
            continue
        if isinstance(value, dict):
            return value
        index = start + 1

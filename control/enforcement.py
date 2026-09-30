"""Dual-side schema enforcement (proposal.tmp Section 7).

Producer side: JSON-mode instruction appended to the agent prompt.
Consumer side: extract + Pydantic-validate; exactly one repair retry,
then the verdict is logged invalid (never silently coerced).
"""

import json
import re

from pydantic import ValidationError

from control.schemas import HAM10000_CLASSES, VerdictEnvelope


JSON_INSTRUCTION = """\
You MUST reply with exactly one JSON object and no other text.
Schema: {"ran": bool, "command": string, "exit_code": int|null, \
"predicted_class": one of %s|null, \
"confidence": number in [0,1]|null, "reasoning": string, \
"uncertainty_flags": subset of ["borderline","no_evidence",\
"tool_failed","disagreement"]}.
If you could not run a tool, set ran=false and explain in reasoning.
Report the exact command you ran plus your reasoning in words.
predicted_class and confidence are filled in control-side from
re-execution: you MAY set them to null. If you report tool numbers,
copy disease_name and confidence_score EXACTLY as the tool printed
them; inventing numbers is the worst failure — omission is safe,
fabrication is not.
""" % (list(HAM10000_CLASSES),)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class EnvelopeError(ValueError):
    """Raised when raw agent output is not a valid verdict envelope."""


def parse_envelope(raw: str) -> VerdictEnvelope:
    match = _JSON_OBJECT_RE.search(raw.strip())
    if not match:
        raise EnvelopeError("No JSON object found in agent output.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise EnvelopeError(f"Malformed JSON: {exc}") from exc
    try:
        return VerdictEnvelope(**payload)
    except ValidationError as exc:
        raise EnvelopeError(f"Schema violation: {exc}") from exc


def build_repair_prompt(raw: str, error: str) -> str:
    return (
        "Your previous reply was rejected for this reason:\n"
        f"{error}\n\nPrevious reply:\n{raw.strip()}\n\n"
        "Reply ONCE more with a corrected single JSON object. "
        + JSON_INSTRUCTION
    )

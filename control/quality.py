"""Quality Agent (plan.tmp §2.1 — chain step 1, LLM thinker).

Toolkit (installed skills): `quality-gate`, `preprocess`,
`abcde-analyzer` (read-only context). The agent decides itself whether
the image is good enough (pass), cleans it with preprocess 0..n times
by its own judgment (cleaned), or passes it forward with flags
(flagged). No blur threshold lives in code.

Output: short JSON + text — `pass|cleaned|flagged` + reasons, citing
the tool JSON it saw. Never diagnoses; never stops the round: a
`flagged` image flows forward with its flag, and a tool crash is
`no-evidence` for that check, never an abort. Invalid output fails
safe to `flagged` (forward with flag), never to a stop.
"""

from __future__ import annotations

import json
import re as _re

QUALITY_TOOLKIT: tuple[str, ...] = (
    "quality-gate",
    "preprocess",
    "abcde-analyzer",
)

QUALITY_SCHEMA = (
    '{"verdict": "pass|cleaned|flagged", "reasons": [string], '
    '"cites": [string], "cleaned_image": string|null}'
)

VALID_VERDICTS = ("pass", "cleaned", "flagged")


def build_quality_prompt(record: dict, role_text: str,
                          toolkit_text: str) -> str:
    """Prompt the Quality Agent with case + role + its toolkit only."""
    metadata = json.dumps(record.get("metadata", {}), ensure_ascii=False)
    return (
        f"{role_text}\n"
        f"Case image: {record.get('image_path')}\n"
        f"Case metadata (context only, never for diagnosis): {metadata}\n"
        f"Your toolkit:\n{toolkit_text}\n\n"
        "Assess this image's quality now: run `quality_gate` first, "
        "clean with `preprocess` if YOU judge it worthwhile (0..n times), "
        "read `abcde_analyzer` only as context. Then report.\n\n"
        "Reply with exactly one JSON object, no other text: "
        + QUALITY_SCHEMA
    )


def parse_quality(raw: str) -> dict:
    """Strict-parse a Quality Agent reply. Never raises.

    Fail-safe is always `flagged` (flows forward with its flag): malformed
    JSON, schema violations, missing cites, or a `cleaned` claim without
    a `cleaned_image` path all become `flagged` with the defect recorded.
    Nothing here may stop the round.
    """

    def flagged(*reasons: str) -> dict:
        return {
            "verdict": "flagged",
            "reasons": [str(r)[:300] for r in reasons] or ["invalid output"],
            "cites": ["fail-safe: unparseable or schema-violating reply"],
            "cleaned_image": None,
        }

    match = _re.search(r"\{.*\}", (raw or "").strip(), _re.DOTALL)
    if not match:
        return flagged("no JSON object in Quality Agent reply")
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return flagged("malformed JSON in Quality Agent reply")
    if not isinstance(payload, dict):
        return flagged("Quality Agent reply is not a JSON object")
    verdict = payload.get("verdict")
    if verdict not in VALID_VERDICTS:
        return flagged(f"invalid verdict {verdict!r}: "
                       "must be pass|cleaned|flagged")
    reasons = payload.get("reasons")
    cites = payload.get("cites")
    if not isinstance(reasons, list) or not reasons or not all(
            isinstance(r, str) and r.strip() for r in reasons):
        return flagged("quality verdict without reasons")
    if not isinstance(cites, list) or not cites or not all(
            isinstance(c, str) and c.strip() for c in cites):
        return flagged("quality verdict without cites")
    cleaned_image = payload.get("cleaned_image")
    if cleaned_image is not None and not (
            isinstance(cleaned_image, str) and cleaned_image.strip()):
        return flagged("invalid cleaned_image: must be a path or null")
    if verdict == "cleaned" and cleaned_image is None:
        return flagged("claimed cleaned without a cleaned_image path")
    return {
        "verdict": verdict,
        "reasons": [str(r)[:300] for r in reasons],
        "cites": [str(c)[:300] for c in cites],
        "cleaned_image": cleaned_image,
    }


def run_quality(record: dict, toolkit_text: str,
                agent_id: str = "main") -> dict:
    """Run one Quality Agent session. Never raises, never stops the round.

    Backend failure, missing role files, or invalid output all become a
    `flagged` verdict that flows forward — the round continues.
    """
    from control.prompts import load_role  # lazy: file layout dependency

    try:
        role_text = load_role("quality")
    except Exception as exc:  # noqa: BLE001 - flag, don't crash
        return {
            "verdict": "flagged",
            "reasons": [f"quality role unavailable: {str(exc)[:150]}"],
            "cites": ["fail-safe: role prompt failed to load"],
            "cleaned_image": None,
        }
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    try:
        response = run_openclaw_cli(
            build_quality_prompt(record, role_text, toolkit_text),
            agent_id=agent_id, show_command=False)
    except Exception as exc:  # noqa: BLE001 - advisory role only
        return {
            "verdict": "flagged",
            "reasons": [f"quality backend unavailable: {str(exc)[:150]}"],
            "cites": ["fail-safe: agent backend failed"],
            "cleaned_image": None,
        }
    if isinstance(response, dict) and response.get("payloads"):
        raw = "\n".join(item.get("text", "")
                         for item in response["payloads"])
    else:
        raw = json.dumps(response, ensure_ascii=False)
    return parse_quality(raw)


def quality_line(quality: dict) -> str:
    """One-line audit/report rendering: verdict + cited reasons."""
    verdict = quality.get("verdict", "flagged")
    reasons = "; ".join(quality.get("reasons", [])[:3])
    cleaned = quality.get("cleaned_image")
    extra = f" → {cleaned}" if verdict == "cleaned" and cleaned else ""
    return f"Quality: {verdict}{extra} — {reasons}"

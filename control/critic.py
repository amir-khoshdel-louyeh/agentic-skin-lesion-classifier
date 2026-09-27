"""Critic role (Phase 10, flagged cases only).

Asymmetric by design: the critic argues ONLY for rejection. It cannot
confirm, clear, or diagnose — a sustained objection forces
suspicious-refer; silence or an invalid reply changes nothing.
"""

from __future__ import annotations

import json

CRITIC_SCHEMA = (
    '{"objection": bool, "reasons": [string], "cites": [string]}'
)


def build_critic_prompt(verdicts: list, flags: list[str]) -> str:
    """Prompt the critic with receipted verdicts + case flags."""
    lines = []
    for i, v in enumerate(verdicts):
        lines.append(
            f"Verdict {i + 1}: class={v.predicted_class}, "
            f"confidence={v.confidence}, "
            f"flags={','.join(v.uncertainty_flags) or 'none'}, "
            f"reasoning={v.reasoning[:300]}"
        )
    return (
        "You are the safety critic of a skin-lesion screening round. "
        "Your ONLY job is to object to unsafe verdicts. You may NOT "
        "confirm, clear, or diagnose any case.\n\n"
        "Object (objection=true) ONLY if you can point to a concrete "
        "defect in the evidence below: a miscounted receipt, a confidence "
        "below its threshold, an unflagged disagreement, or a case flag "
        "the verdicts ignore. Each reason MUST cite the verdict number or "
        "flag it relies on.\n\n"
        f"Case flags: {', '.join(flags) or 'none'}\n"
        + "\n".join(lines)
        + "\n\nReply with exactly one JSON object, no other text: "
        + CRITIC_SCHEMA
    )


def parse_objection(raw: str, verdict_count: int) -> tuple[bool, list[str]]:
    """Strict-parse a critic reply. Returns (sustained, reasons).

    Fail-safe toward silence: anything invalid, uncited, or out of range
    is NOT an objection. A broken critic must never force referrals by
    itself — the calibrated verdicts already do that when unsure.
    """
    import re as _re

    match = _re.search(r"\{.*\}", raw.strip(), _re.DOTALL)
    if not match:
        return False, ["no JSON object"]
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return False, ["malformed JSON"]
    if not isinstance(payload, dict) or payload.get("objection") is not True:
        return False, ["no objection raised"]
    reasons = payload.get("reasons")
    cites = payload.get("cites")
    if not isinstance(reasons, list) or not reasons:
        return False, ["objection without reasons"]
    if not isinstance(cites, list) or not cites:
        return False, ["objection without cites"]
    for cite in cites:
        nums = [int(n) for n in _re.findall(r"\d+", str(cite))]
        if nums and not any(1 <= n <= verdict_count for n in nums):
            return False, [f"cite out of range: {cite!r}"]
    return True, [str(r)[:300] for r in reasons]


def run_critic(verdicts: list, flags: list[str],
               agent_id: str = "main") -> tuple[bool, list[str]]:
    """Run one critic session. Never raises: backend failure = silence."""
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    try:
        response = run_openclaw_cli(build_critic_prompt(verdicts, flags),
                                    agent_id=agent_id, show_command=False)
    except Exception as exc:  # noqa: BLE001 - critic is advisory only
        return False, [f"critic backend unavailable: {str(exc)[:150]}"]
    if isinstance(response, dict) and response.get("payloads"):
        raw = "\n".join(item.get("text", "")
                        for item in response["payloads"])
    else:
        raw = json.dumps(response, ensure_ascii=False)
    return parse_objection(raw, len(verdicts))

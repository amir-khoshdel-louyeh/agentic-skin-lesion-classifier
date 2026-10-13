"""Critic role (Phase 10, flagged cases only).

Asymmetric by design: the critic argues ONLY for rejection. It cannot
confirm, clear, or diagnose — a sustained objection forces
suspicious-refer; silence or an invalid reply changes nothing.
"""

from __future__ import annotations

import json

CRITIC_SCHEMA = (
    '{"objection": bool, '
    '"code": "unflagged_disagreement|miscounted_receipt", '
    '"reasons": [string], "cites": [string]}'
)

# Closed set of defect classes the referee does NOT already handle. Image
# quality is a machine rule now, and confidence thresholds are counted by
# decide(), so an objection to either is not actionable — the model used
# to raise it every time on a flagged case, which turned the referee into
# a rubber stamp. A code is only honored when the evidence matches it.
OBJECTION_CODES = ("unflagged_disagreement", "miscounted_receipt")


def build_critic_prompt(verdicts: list, flags: list[str],
                          quality_text: str = "") -> str:
    """Prompt the critic with receipted verdicts + case flags + Quality.

    Plan.tmp §2.4 toolkit: read-only receipts + Quality flags. The critic
    sees re-executed numbers only, never agent prose claims.
    """
    lines = []
    for i, v in enumerate(verdicts):
        lines.append(
            f"Verdict {i + 1}: class={v.predicted_class}, "
            f"confidence={v.confidence}, "
            f"flags={','.join(v.uncertainty_flags) or 'none'}, "
            f"reasoning={v.reasoning[:300]}"
        )
    quality_section = (f"Quality Agent (chain step 1): {quality_text}\n"
                       if quality_text else "")
    return (
        "You are the safety critic of a skin-lesion screening round. "
        "Your ONLY job is to object to unsafe verdicts. You may NOT "
        "confirm, clear, or diagnose any case.\n\n"
        "Image quality is ALREADY handled deterministically by the "
        "referee: a flagged image can never be reported benign. Do NOT "
        "object merely because the image is flagged, and do NOT object "
        "because a diagnostic verdict's flags list is empty — verdict "
        "flags come from the model tool itself (entropy gate), never from "
        "the separate Quality Agent step.\n\n"
        "Object (objection=true) ONLY for a concrete defect INSIDE the "
        "verdicts below, and name its code:\n"
        "- \"unflagged_disagreement\": two countable verdicts disagree and "
        "neither carries a disagreement flag.\n"
        "- \"miscounted_receipt\": a verdict was counted without a valid "
        "receipt.\n"
        "Anything else (blur, quality, low confidence) has no code: the "
        "referee already handles it deterministically. Each reason MUST "
        "cite the verdict number it relies on.\n\n"
        f"{quality_section}"
        f"Case flags: {', '.join(flags) or 'none'}\n"
        + "\n".join(lines)
        + "\n\nReply with exactly one JSON object, no other text: "
        + CRITIC_SCHEMA
    )


def parse_objection(raw: str, verdict_count: int,
                    verdicts: list | None = None
                    ) -> tuple[bool, list[str]]:
    """Strict-parse a critic reply. Returns (sustained, reasons).

    Fail-safe toward silence: anything invalid, uncited, or out of range
    is NOT an objection. A broken critic must never force referrals by
    itself — the calibrated verdicts already do that when unsure.

    When `verdicts` is given, the claimed defect code is checked against
    the evidence it names: an `unflagged_disagreement` needs two distinct
    countable classes, and a `miscounted_receipt` needs a non-countable
    verdict. A code that the evidence contradicts is silence.
    """
    import re as _re

    from control.jsontext import first_json_object

    payload = first_json_object(raw)
    if payload is None:
        return False, ["no JSON object"]
    if payload.get("objection") is not True:
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
    code = payload.get("code")
    if code not in OBJECTION_CODES:
        return False, [f"objection without an actionable code: {code!r}"]
    if verdicts is not None:
        countable = [v for v in verdicts if getattr(v, "is_countable",
                                                lambda: False)()]
        if code == "unflagged_disagreement":
            classes = {v.predicted_class for v in countable}
            if len(classes) < 2:
                return False, ["unflagged_disagreement claimed but every "
                               "countable verdict agrees"]
        if code == "miscounted_receipt" and len(countable) == len(verdicts):
            return False, ["miscounted_receipt claimed but every receipt is "
                           "valid"]
    return True, [str(r)[:300] for r in reasons]


def run_critic(verdicts: list, flags: list[str],
               agent_id: str = "main",
               quality_text: str = "") -> tuple[bool, list[str]]:
    """Run one critic session. Never raises: backend failure = silence."""
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    try:
        response = run_openclaw_cli(
            build_critic_prompt(verdicts, flags, quality_text),
            agent_id=agent_id, show_command=False)
    except Exception as exc:  # noqa: BLE001 - critic is advisory only
        return False, [f"critic backend unavailable: {str(exc)[:150]}"]
    if isinstance(response, dict) and response.get("payloads"):
        raw = "\n".join(item.get("text", "")
                        for item in response["payloads"])
    else:
        raw = json.dumps(response, ensure_ascii=False)
    try:
        from control.vram import unload_ollama as _unload
        _unload()  # sequential GPU: free LLM VRAM before tool phases
    except Exception:
        pass
    return parse_objection(raw, len(verdicts), verdicts=verdicts)

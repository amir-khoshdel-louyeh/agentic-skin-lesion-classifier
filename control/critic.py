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

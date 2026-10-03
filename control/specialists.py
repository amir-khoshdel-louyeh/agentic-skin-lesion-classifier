"""Disease specialists (Phase 11, disagreement only).

When countable votes scatter, compute is spent on the actual dilemma:
top-2 diseases by calibrated confidence. Max 2 advocates, sequential.
Each advocate is constrained to ONE assigned disease, uses the same
envelope schema, and goes through the same receipt verification.
"""

from __future__ import annotations


def shortlist(verdicts: list, limit: int = 2) -> list[str] | None:
    """Top-`limit` distinct predicted classes by confidence.

    Returns None when there is no dilemma (< 2 distinct countable
    classes). A lone disagreement flag without a second class is not a
    dilemma — there is nothing to advocate against.
    """
    best: dict[str, float] = {}
    for v in verdicts:
        if not v.is_countable():
            continue
        conf = v.confidence if v.confidence is not None else 0.0
        if v.predicted_class not in best or conf > best[v.predicted_class]:
            best[v.predicted_class] = conf
    if len(best) < 2:
        return None
    ranked = sorted(best, key=lambda c: best[c], reverse=True)
    return ranked[:limit]


def specialist_brief(disease: str, shortlisted: list[str]) -> str:
    """Role assignment: advocate ONLY for `disease`.

    Static fallback template (plan.tmp T7): used only when the Brain
    cannot write angles. The preferred path is advocate_brief() with a
    Brain-written angle.
    """
    others = ", ".join(c for c in shortlisted if c != disease) or "none"
    return (
        f"Specialist role: advocate ONLY for {disease}. Use any ONE ready "
        f"diagnostic tool with the case image and metadata. Argue why the "
        f"evidence fits {disease} (the alternative under debate: {others}). "
        f"Your predicted_class MUST be {disease} unless the tool's own "
        f"output says otherwise — report the tool output faithfully, "
        f"never fabricate it. Cite the receipt command and numbers."
    )


def advocate_brief(disease: str, shortlisted: list[str],
                   angle: str) -> str:
    """Advocate assignment with a Brain-written angle (plan.tmp T7)."""
    return specialist_brief(disease, shortlisted) + (
        f" Brain angle for your advocacy (steering only, verify it "
        f"yourself): {angle}")


def confine_specialist(verified, disease: str,
                       shortlisted: list[str]):
    """Off-shortlist filter (plan.tmp T7/T11, counting only).

    A specialist confined to `disease` may not introduce a third
    candidate: an off-shortlist countable class becomes no-evidence
    (ran=False). On-shortlist verdicts pass through untouched.
    """
    from control.schemas import VerdictEnvelope  # lazy: schema owner

    if verified.is_countable() and verified.predicted_class != disease:
        return VerdictEnvelope(
            ran=False,
            command=verified.command,
            reasoning=(f"Specialist off-shortlist ({verified.predicted_class} "
                       f"not in {shortlisted}) — recorded as no evidence."),
            uncertainty_flags=["tool_failed"],
        )
    return verified

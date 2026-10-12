"""Consensus entropy gate for the diagnostic tools (proposal.tmp §7).

The tools stack their own softmax vectors and gate the consensus with
this helper: a distribution whose entropy exceeds the threshold is
marked borderline, and a top class disagreeing with a tool vote adds a
disagreement flag. Numbers still come from re-executed tool output, not
the LLM.

Retired here: the earlier Brio-style *closed-set LLM re-query* (ask the
agent for 7-class probabilities via logprobs). It was never wired into
a round and would reintroduce LLM-authored numbers, which the
single-writer protocol forbids, so the unused prompt builder is gone.
"""

import math
import os

DEFAULT_ENTROPY_THRESHOLD = 0.9  # nats; max ln(7) ~= 1.95

# The doctor file's `entropy_threshold` used to be print-only: every tool
# hardcoded the default, so a physician's edit changed the report text and
# nothing else. The orchestrator exports the validated value here and every
# gate reads it, so the edit actually takes effect.
ENTROPY_ENV = "SKIN_ENTROPY_THRESHOLD"


def env_entropy_threshold(default: float = DEFAULT_ENTROPY_THRESHOLD) -> float:
    """Validated entropy threshold from the environment (or the default)."""
    raw = os.environ.get(ENTROPY_ENV)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return default
    # Outside the doctor's allowed band the value is ignored, never clamped
    # silently into a laxer gate.
    return value if 0.0 < value <= math.log(7) else default


def entropy(probs: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0.0)


def evaluate(
    probs: dict[str, float],
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    tool_class: str | None = None,
) -> dict[str, object]:
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError("Empty probability distribution.")
    norm = {k: v / total for k, v in probs.items()}
    top_class = max(norm, key=norm.get)
    confidence = norm[top_class]
    ent = entropy(norm)
    flags: list[str] = []
    if ent > entropy_threshold:
        flags.append("borderline")
    if tool_class is not None and tool_class != top_class:
        flags.append("disagreement")
    return {
        "top_class": top_class,
        "confidence": round(confidence, 4),
        "entropy": round(ent, 4),
        "uncertainty_flags": flags,
    }

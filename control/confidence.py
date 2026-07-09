"""Closed-set confidence protocol, Brio-style (proposal.tmp Section 7).

Instead of trusting a generated confidence number, the agent is queried
a second time with the 7 HAM10000 classes as the ONLY allowed options
(Ollama JSON mode + logprobs). Reported confidence = top-option
probability; entropy above threshold forces borderline automatically.
"""

import math

from control.schemas import HAM10000_CLASSES

DEFAULT_ENTROPY_THRESHOLD = 0.9  # nats; max ln(7) ~= 1.95


def build_closed_set_prompt(case_summary: str) -> str:
    options = "\n".join(f"- {name}" for name in HAM10000_CLASSES)
    return (
        'Classify the lesion. Reply with exactly one JSON object, no other '
        'text: {"probabilities": {<class>: <probability 0..1>}, '
        '"reasoning": string}. The class MUST be one of:\n'
        + options
        + '\nProbabilities must sum to 1. Case:\n'
        + case_summary.strip()
    )


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

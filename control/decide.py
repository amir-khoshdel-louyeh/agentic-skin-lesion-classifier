"""Decision core: vote, escalate, report, audit (proposal.tmp Sec 4/8).

Rules (locked layer, not doctor-editable):
- Only receipted verdicts count (command + exit_code 0 + known class).
- ANY melanoma vote at/above threshold OR ANY borderline flag forces
  "suspicious — refer to physician", never a confident benign.
- Zero countable verdicts => "inconclusive — no evidence".
- Otherwise weighted majority wins.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from control.paths import ROOT_DIR
from control.schemas import VerdictEnvelope

MALIGNANT_CLASS = "Melanoma"
DECISIONS = ("benign", "suspicious", "inconclusive")


def decide(
    verdicts: list[VerdictEnvelope],
    weights: dict[int, float] | None = None,
    borderline_threshold: float = 0.75,
    critic_objection: bool = False,
) -> dict[str, object]:
    countable = [v for v in verdicts if v.is_countable()]
    if not countable:
        return {
            "decision": "inconclusive",
            "reason": "no receipted verdicts",
            "verdicts_counted": 0,
        }
    if critic_objection:
        return {
            "decision": "suspicious",
            "reason": "sustained critic objection",
            "verdicts_counted": len(countable),
        }
    for v in countable:
        if v.predicted_class == MALIGNANT_CLASS and (v.confidence or 0.0) >= borderline_threshold:
            return {
                "decision": "suspicious",
                "reason": f"malignancy vote ({v.confidence})",
                "verdicts_counted": len(countable),
            }
        if "borderline" in v.uncertainty_flags or (v.confidence or 0.0) < borderline_threshold:
            return {
                "decision": "suspicious",
                "reason": f"borderline verdict ({v.predicted_class}, {v.confidence})",
                "verdicts_counted": len(countable),
            }
    tally: dict[str, float] = {}
    for i, v in enumerate(countable):
        w = (weights or {}).get(i, 1.0)
        tally[v.predicted_class] = tally.get(v.predicted_class, 0.0) + w
    winner = max(tally, key=tally.get)
    return {
        "decision": "benign",
        "reason": f"weighted majority: {winner} ({tally[winner]})",
        "verdicts_counted": len(countable),
        "final_class": winner,
    }


def build_report(
    record: dict,
    verdicts: list[VerdictEnvelope],
    decision: dict,
    thresholds: dict[str, float] | None = None,
) -> str:
    lines = [
        "# Medical Screening Report (research use only)",
        "",
        f"- Image: {record.get('image_path')}",
        f"- Metadata: {json.dumps(record.get('metadata', {}), ensure_ascii=False)}",
        f"- Decision: **{decision['decision']}** — {decision['reason']}",
    ]
    if thresholds:
        lines.append(
            "- Thresholds: "
            + ", ".join(f"{k}={v}" for k, v in thresholds.items())
        )
    lines += [
        "",
        "## Agent verdicts",
        "",
    ]
    for i, v in enumerate(verdicts):
        lines += [
            f"### Agent {i + 1}",
            f"- ran: {v.ran}, exit: {v.exit_code}, command: `{v.command}`",
            f"- class: {v.predicted_class}, confidence: {v.confidence}",
            f"- flags: {', '.join(v.uncertainty_flags) or 'none'}",
            f"- reasoning: {v.reasoning}",
            "",
        ]
    lines += [
        "Coverage: 7 HAM10000 classes only — squamous cell carcinoma (SCC) "
        "and other conditions are out of coverage; flat uncertain cases "
        "are referred, never forced into a near class.",
        "Research use only — not a medical diagnosis.",
        "",
    ]
    return "\n".join(lines)


def append_audit(entry: dict, root: Path = ROOT_DIR) -> Path:
    path = root / "report" / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry, ts=datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path

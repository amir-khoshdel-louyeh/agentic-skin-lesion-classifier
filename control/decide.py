"""Decision core: vote, escalate, report, audit (proposal.tmp Sec 4/8).

Rules (locked layer, not doctor-editable):
- Only receipted verdicts count (command + exit_code 0 + known class).
- ANY melanoma vote at/above threshold OR ANY borderline flag forces
  "suspicious — refer to physician", never a confident benign.
- Zero countable verdicts => "inconclusive — no evidence".
- Otherwise weighted majority wins.
"""

import json
import re as _re
from datetime import datetime, timezone
from pathlib import Path

from control.paths import ROOT_DIR
from control.schemas import VerdictEnvelope

MALIGNANT_CLASS = "Melanoma"
DECISIONS = ("benign", "suspicious", "inconclusive")

# Referee rules (plan.tmp §7 Part A formula: which rule fired).
RULE_NO_EVIDENCE = "no_evidence"
RULE_CRITIC = "critic_objection"
RULE_MALIGNANCY = "malignancy_vote"
RULE_BORDERLINE = "borderline_flag"
RULE_MAJORITY = "weighted_majority"
RULE_DILEMMA = "specialist_dilemma"


def reliability_of(reasoning: str) -> str:
    """Reliability tag from a re-executed verdict note (informational).

    The tag is written control-side by verify_receipt; anything without a
    tag (e.g. careful-path extras, legacy notes) reports "unknown".
    Never a vote gate — Part A display only.
    """
    match = _re.search(r"reliability=(exact|rounded|fabricated|omitted)",
                       reasoning or "")
    return match.group(1) if match else "unknown"


def receipt_tool(command: str) -> str:
    """Tool short name from a verified command, or the raw command."""
    parts = (command or "").split()
    if len(parts) >= 2 and parts[0] == "python" and parts[1].startswith(
            "tools/") and parts[1].endswith(".py"):
        return parts[1][len("tools/"):-len(".py")]
    return command or "—"


def decide(
    verdicts: list[VerdictEnvelope],
    weights: dict[int, float] | None = None,
    borderline_threshold: float = 0.75,
    critic_objection: bool = False,
) -> dict[str, object]:
    # plan.tmp T5: single-writer lock. ONLY re-executed VerdictEnvelopes
    # may vote. Anything else (e.g. a Quality Agent dict, critic text,
    # raw tool JSON) fails fast instead of silently counting — advisory
    # voices never become votes by accident.
    for v in verdicts:
        if not isinstance(v, VerdictEnvelope):
            raise TypeError(
                f"decide() takes VerdictEnvelope only, got "
                f"{type(v).__name__}: advisory output is never countable.")
    countable = [v for v in verdicts if v.is_countable()]
    if not countable:
        return {
            "decision": "inconclusive",
            "reason": "no receipted verdicts",
            "rule": RULE_NO_EVIDENCE,
            "verdicts_counted": 0,
        }
    if critic_objection:
        return {
            "decision": "suspicious",
            "reason": "sustained critic objection",
            "rule": RULE_CRITIC,
            "verdicts_counted": len(countable),
        }
    for v in countable:
        if v.predicted_class == MALIGNANT_CLASS and (v.confidence or 0.0) >= borderline_threshold:
            return {
                "decision": "suspicious",
                "reason": f"malignancy vote ({v.confidence})",
                "rule": RULE_MALIGNANCY,
                "verdicts_counted": len(countable),
            }
        if "borderline" in v.uncertainty_flags or (v.confidence or 0.0) < borderline_threshold:
            return {
                "decision": "suspicious",
                "reason": f"borderline verdict ({v.predicted_class}, {v.confidence})",
                "rule": RULE_BORDERLINE,
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
        "rule": RULE_MAJORITY,
        "verdicts_counted": len(countable),
        "final_class": winner,
    }


PART_B_HEADER = "## Part B — LLM voices (explanatory, zero vote power)"


def build_report(
    record: dict,
    verdicts: list[VerdictEnvelope],
    decision: dict,
    thresholds: dict[str, float] | None = None,
    references: list[str] | None = None,
    quality: dict | None = None,
    steers: list[dict] | None = None,
    sessions: list[dict] | None = None,
    critic: dict | None = None,
    specialist_texts: list[dict] | None = None,
    qa: list[dict] | None = None,
    include_part_b: bool = True,
) -> str:
    """Two-part round report (plan.tmp §7).

    Part A (machine, authoritative): receipt table with RE-EXECUTED
    numbers only, Quality line, referee line with the fired rule. Nothing
    in Part A comes from LLM prose — deleting all of Part B leaves the
    decision identical (delete-all-prose test).
    Part B (LLM voices, zero vote power): steers, session reasoning,
    critic, specialists, Q&A annex.
    """
    lines = [
        "# Medical Screening Report (research use only)",
        "",
        f"- Image: {record.get('image_path')}",
        f"- Metadata: {json.dumps(record.get('metadata', {}), ensure_ascii=False)}",
        "",
        "## Part A — Decision source (machine, authoritative)",
        "",
        "### Receipts (re-executed numbers only)",
        "",
        "| # | tool | class | confidence | flags | reliability |",
        "|---|------|-------|------------|-------|-------------|",
    ]
    for i, v in enumerate(verdicts):
        lines.append(
            f"| {i + 1} | {receipt_tool(v.command)} | "
            f"{v.predicted_class or '—'} | "
            f"{v.confidence if v.confidence is not None else '—'} | "
            f"{','.join(v.uncertainty_flags) or 'none'} | "
            f"{reliability_of(v.reasoning)} |"
        )
    lines += ["", "### Quality"]
    if quality:
        cleaned = (f" → {quality['cleaned_image']}"
                   if quality.get("verdict") == "cleaned"
                   and quality.get("cleaned_image") else "")
        cites = "; ".join(quality.get("cites", [])[:3])
        lines += [
            "",
            f"- Quality: **{quality.get('verdict', 'flagged')}**{cleaned}",
            f"- Reasons: {'; '.join(quality.get('reasons', []))}",
            f"- Cites: {cites or 'none'}",
        ]
    else:
        lines += ["", "- Quality: not recorded this round."]
    lines += [
        "",
        "### Referee",
        "",
        f"- Decision: **{decision['decision']}** — {decision['reason']}",
        f"- Formula: rule={decision.get('rule', 'unknown')} "
        f"(verdicts counted: {decision.get('verdicts_counted', 0)})",
    ]
    if thresholds:
        lines.append(
            "- Thresholds in force: "
            + ", ".join(f"{k}={v}" for k, v in thresholds.items())
        )
    part_a = "\n".join(lines)
    if not include_part_b:
        return part_a + "\n"
    out = [part_a, PART_B_HEADER, "",
           "> Part B explains; Part A decides. "
           "Part B cannot change Part A.", ""]
    out += ["### Brain steers", ""]
    if steers:
        for s in steers:
            stopped = " [STOP]" if s.get("stopped") or s.get("stop") else ""
            out += [f"{s.get('session', '?')}. {s.get('steer') or '(stop)'}"
                    f"{stopped}",
                    f"   why: {s.get('why', '')}"]
    else:
        out += ["- (none recorded)"]
    out += ["", "### Session reasoning", ""]
    if sessions:
        for s in sessions:
            out += [f"- Session {s.get('n')}: "
                    f"tools={s.get('tools_run', '?')}, "
                    f"class={s.get('class')}, "
                    f"confidence={s.get('confidence')}",
                    f"  reasoning: {s.get('reasoning', '')[:800]}"]
    else:
        for i, v in enumerate(verdicts):
            out += [f"- Execution {i + 1}: {v.reasoning[:800]}"]
    out += ["", "### Critic", ""]
    if critic and critic.get("sustained"):
        out += ["- OBJECTION sustained:",
                *[f"  - {r}" for r in critic.get("reasons", [])]]
    else:
        out += ["- silence."]
    out += ["", "### Specialists", ""]
    if specialist_texts:
        for sp in specialist_texts:
            out += [f"- Advocate for {sp.get('disease')}:",
                    f"  {sp.get('reasoning', '')[:800]}"]
    else:
        out += ["- (none — no dilemma this round)"]
    if qa:
        out += ["", "### Q&A annex", ""]
        for item in qa:
            out += [f"- Q: {item.get('question', '')}",
                    f"  A: {item.get('answer', '')[:800]}"]
    out += [
        "",
        "## References",
        "",
    ]
    if references:
        out += [f"- {ref}" for ref in references] + [""]
    else:
        out += ["- Local tools only — no web sources consulted.", ""]
    out += [
        "Coverage: 7 HAM10000 classes only — squamous cell carcinoma (SCC) "
        "and other conditions are out of coverage; flat uncertain cases "
        "are referred, never forced into a near class.",
        "Research use only — not a medical diagnosis.",
        "",
    ]
    return "\n".join(out)


def append_audit(entry: dict, root: Path = ROOT_DIR) -> Path:
    path = root / "report" / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry, ts=datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path

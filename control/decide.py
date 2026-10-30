"""Decision core: vote, escalate, report, audit (proposal.tmp Sec 4/8).

Rules (locked layer, not doctor-editable):
- Only receipted verdicts count (command + exit_code 0 + known class).
- ANY melanoma vote at/above threshold, ANY tool borderline flag, ANY
  confidence below the confidence floor, or an unresolved quality flag
  forces "suspicious — refer to physician", never a confident benign.
- Zero countable verdicts => "inconclusive — no evidence".
- Otherwise weighted majority wins. One vote per TOOL: re-running the
  same model is not independent confirmation.
- Machine rules run before the critic, so Part A always names a
  reproducible reason when one exists.
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
RULE_QUALITY = "quality_flagged"
RULE_CONFIDENCE = "confidence_floor"


def reliability_of(reasoning: str) -> str:
    """Reliability tag from a re-executed verdict note (informational).

    The tag is written control-side by verify_receipt; anything without a
    tag (e.g. careful-path extras, legacy notes) reports "unknown".
    Never a vote gate — Part A display only.
    """
    match = _re.search(
        r"reliability=(exact|rounded|fabricated|omitted|class_mismatch)",
        reasoning or "")
    return match.group(1) if match else "unknown"


def receipt_tool(command: str) -> str:
    """Tool short name from a verified command, or the raw command."""
    parts = (command or "").split()
    if len(parts) >= 2 and parts[0] == "python" and parts[1].startswith(
            "tools/") and parts[1].endswith(".py"):
        return parts[1][len("tools/"):-len(".py")]
    return command or "—"


def receipt_ledger_row(verdict: VerdictEnvelope) -> dict:
    """Audit/report projection of one content-addressed execution receipt."""
    return {
        "receipt_id": verdict.receipt_id,
        "tool": verdict.tool or receipt_tool(verdict.command),
        "command": verdict.command,
        "command_sha256": verdict.command_sha256,
        "image_sha256": verdict.image_sha256,
        "stdout_sha256": verdict.stdout_sha256,
        "exit_code": verdict.exit_code,
        "duration_ms": verdict.duration_ms,
    }


def decide(
    verdicts: list[VerdictEnvelope],
    weights: dict[int, float] | None = None,
    borderline_threshold: float = 0.75,
    confidence_threshold: float = 0.75,
    critic_objection: bool = False,
    quality_flagged: bool = False,
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
            "distinct_tools": 0,
        }
    # One vote per TOOL, not per session. Running the same model twice must
    # never look like two independent confirmations (it used to: two rows
    # of isic_vit counted as "2 verdicts").
    best: dict[str, VerdictEnvelope] = {}
    for v in countable:
        tool = receipt_tool(v.command)
        if tool not in best or (v.confidence or 0.0) > (best[tool].confidence or 0.0):
            best[tool] = v
    unique = list(best.values())
    counted = len(countable)
    # Machine rules run FIRST and in order of specificity. The critic is an
    # LLM voice: it may only raise suspicion when no deterministic rule
    # already fired, so Part A always reports a reproducible reason.
    for v in unique:
        if v.predicted_class == MALIGNANT_CLASS and (v.confidence or 0.0) >= borderline_threshold:
            return {
                "decision": "suspicious",
                "reason": f"malignancy vote ({v.confidence})",
                "rule": RULE_MALIGNANCY,
                "verdicts_counted": counted,
                "distinct_tools": len(unique),
            }
        if "borderline" in v.uncertainty_flags or (v.confidence or 0.0) < borderline_threshold:
            return {
                "decision": "suspicious",
                "reason": f"borderline verdict ({v.predicted_class}, {v.confidence})",
                "rule": RULE_BORDERLINE,
                "verdicts_counted": counted,
                "distinct_tools": len(unique),
            }
    # confidence_threshold is the minimum confidence a BENIGN conclusion
    # may rest on. Raising it in the doctor file flags more cases, which
    # is what prompts.py promises; at the default it coincides with the
    # borderline rule above and changes nothing.
    weak = [v for v in unique
            if v.confidence is not None and v.confidence < confidence_threshold]
    if weak:
        worst = min(weak, key=lambda v: v.confidence or 0.0)
        return {
            "decision": "suspicious",
            "reason": (f"confidence below floor ({worst.predicted_class}, "
                       f"{worst.confidence} < {confidence_threshold})"),
            "rule": RULE_CONFIDENCE,
            "verdicts_counted": counted,
            "distinct_tools": len(unique),
        }
    # A flagged image may never end as "benign": the quality verdict is a
    # machine measurement, so it binds here instead of relying on an LLM
    # critic to notice it. Malignancy/borderline rules above keep their
    # more specific reasons when they fire first.
    if quality_flagged:
        return {
            "decision": "suspicious",
            "reason": ("image quality unresolved (flagged) — re-capture or "
                       "review before relying on this screening"),
            "rule": RULE_QUALITY,
            "verdicts_counted": counted,
            "distinct_tools": len(unique),
        }
    if critic_objection:
        return {
            "decision": "suspicious",
            "reason": "sustained critic objection",
            "rule": RULE_CRITIC,
            "verdicts_counted": counted,
            "distinct_tools": len(unique),
        }
    tally: dict[str, float] = {}
    for i, v in enumerate(unique):
        # Weights are keyed by the deduped position: a tool's evidence is
        # counted once, no matter how many sessions re-ran it.
        w = (weights or {}).get(i, 1.0)
        tally[v.predicted_class] = tally.get(v.predicted_class, 0.0) + w
    winner = max(tally, key=tally.get)
    return {
        "decision": "benign",
        "reason": f"weighted majority: {winner} ({tally[winner]})",
        "rule": RULE_MAJORITY,
        "verdicts_counted": counted,
        "distinct_tools": len(unique),
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
    provenance: dict | None = None,
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
    ]
    if provenance:
        if provenance.get("image_sha256"):
            lines.append(f"- Image sha256: {provenance['image_sha256']}")
        if provenance.get("backend"):
            lines.append(f"- Backend: {provenance['backend']}")
        if provenance.get("started_at"):
            lines.append(f"- Round started: {provenance['started_at']}")
        if provenance.get("run_id"):
            lines.append(f"- Run id: {provenance['run_id']}")
    lines += [
        "",
        "## Part A — Decision source (machine, authoritative)",
        "",
        "### Receipts (re-executed numbers only)",
        "",
        "| # | tool | class | confidence | exit | ms | flags (tool) | flags (agent) | reliability | receipt |",
        "|---|------|-------|------------|------|----|--------------|---------------|-------------|---------",
    ]
    for i, v in enumerate(verdicts):
        agent_flags = ",".join(str(f)[:40] for f in v.agent_flags) or "none"
        lines.append(
            f"| {i + 1} | {receipt_tool(v.command)} | "
            f"{v.predicted_class or '—'} | "
            f"{v.confidence if v.confidence is not None else '—'} | "
            f"{v.exit_code if v.exit_code is not None else '—'} | "
            f"{v.duration_ms if v.duration_ms is not None else '—'} | "
            f"{','.join(v.uncertainty_flags) or 'none'} | "
            f"{agent_flags} | "
            f"{reliability_of(v.reasoning)} | "
            f"{v.receipt_id or '—'} |"
        )
    tools_seen = [receipt_tool(v.command) for v in verdicts
                  if v.is_countable()]
    if len(tools_seen) > len(set(tools_seen)):
        lines += ["", "- Note: repeated rows come from the same tool and add "
                  "no independent evidence; the referee counts each tool once."]
    receipted = [v for v in verdicts if v.receipt_id]
    lines += ["", "### Receipt ledger (content hashes)", ""]
    if receipted:
        lines += [
            "| receipt | tool | exit | ms | command sha256 | image sha256 | stdout sha256 |",
            "|---|---|------|----|----------------|--------------|----------------|",
        ]
        for v in receipted:
            lines.append(
                f"| {v.receipt_id} | {v.tool or receipt_tool(v.command)} | "
                f"{v.exit_code if v.exit_code is not None else '—'} | "
                f"{v.duration_ms if v.duration_ms is not None else '—'} | "
                f"{v.command_sha256 or '—'} | {v.image_sha256 or '—'} | "
                f"{v.stdout_sha256 or '—'} |"
            )
        lines.append(
            "- Receipt id = sha256(tool, exact executed command, image sha256, "
            "stdout sha256, exit code)."
        )
    else:
        lines += ["- No execution receipts this round."]
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
            f"- Cites (agent): {cites or 'none'}",
        ]
        control_cites = quality.get("control_cites") or []
        if control_cites:
            lines.append(f"- Gate (control): {'; '.join(control_cites[:3])}")
        if quality.get("cleaned_rejected"):
            lines.append(f"- Cleaned rejected: {quality['cleaned_rejected']}")
    else:
        lines += ["", "- Quality: not recorded this round."]
    lines += [
        "",
        "### Referee",
        "",
        f"- Decision: **{decision['decision']}** — {decision['reason']}",
        f"- Formula: rule={decision.get('rule', 'unknown')} "
        f"(verdicts counted: {decision.get('verdicts_counted', 0)}"
        + (f", distinct tools: {decision['distinct_tools']}"
           if "distinct_tools" in decision else "")
        + ")",
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

"""Stateful screening controller (Phase 9).

Persistence lives HERE — plain code plus evidence files — never in LLM
context. A `CaseState` owns one case: its round report, audit receipts,
decision, and thresholds. It survives across questions and dies only
when the user closes it. Every LLM call built from it is a fresh,
short, evidence-grounded session (see `ask()`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from control.chat import audit_for_record, latest_report
from control.paths import ROOT_DIR


@dataclass
class CaseState:
    record_index: int
    report_name: str = ""
    report_text: str = ""
    receipts: list[dict] = field(default_factory=list)
    decision: str = "unknown"
    reason: str = ""
    thresholds: dict = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return bool(self.report_text)


def parse_decision(report_text: str) -> tuple[str, str]:
    import re

    match = re.search(r"- Decision:\s*\*\*(\w+)\*\*\s*[—-]\s*(.+)",
                      report_text)
    if match:
        return match.group(1).strip().lower(), match.group(2).strip()
    return "unknown", ""


def load_case(record_index: int, root: Path = ROOT_DIR) -> CaseState:
    """Load stored evidence for a case. Raises FileNotFoundError when no
    round has been run yet — the caller must say so, never invent one."""
    report = latest_report(record_index, root)
    if report is None:
        raise FileNotFoundError(
            f"No round report for record {record_index}. "
            f"Run the round first: main.py --record-index {record_index}"
        )
    text = report.read_text(encoding="utf-8")
    decision, reason = parse_decision(text)
    thresholds: dict = {}
    for entry in audit_for_record(record_index, root):
        if isinstance(entry.get("thresholds"), dict):
            thresholds = entry["thresholds"]
    return CaseState(
        record_index=record_index,
        report_name=report.name,
        report_text=text,
        receipts=audit_for_record(record_index, root),
        decision=decision,
        reason=reason,
        thresholds=thresholds,
    )


EVIDENCE_RULE = (
    "Answer ONLY from the evidence below. Cite the receipt or report "
    "line for every factual claim (class, confidence, flag, threshold). "
    "If the question goes beyond the evidence, say exactly: "
    "\"Not in the record — missing evidence: <what>.\" "
    "Never reinterpret tool numbers, never invent a diagnosis, and never "
    "change the recorded decision. You are narrating tools that already "
    "ran; you did not diagnose anything."
)


def build_question_prompt(state: CaseState, question: str) -> str:
    """Fresh-session prompt: stored evidence + one question + the rule."""
    if not state.loaded:
        raise ValueError("CaseState is empty — load_case() first.")
    receipts = "\n".join(
        json.dumps(e, ensure_ascii=False) for e in state.receipts
    )
    return (
        "You are the screening controller discussing a completed round "
        "with the physician.\n\n"
        f"{EVIDENCE_RULE}\n\n"
        f"--- ROUND REPORT ({state.report_name}) ---\n"
        f"{state.report_text}\n"
        f"--- AUDIT RECEIPTS ---\n{receipts or '(none)'}\n\n"
        f"Physician question: {question.strip()}\n"
    )


def ask(state: CaseState, question: str, agent_id: str = "main") -> str:
    """Answer one question in a FRESH short session grounded in `state`.

    No context persists between calls — continuity comes from the stored
    evidence, never from LLM memory. Returns answer text, or a clean
    error string when the agent backend is unavailable.
    """
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    prompt = build_question_prompt(state, question)
    try:
        response = run_openclaw_cli(prompt, agent_id=agent_id,
                                    show_command=False)
    except Exception as exc:  # noqa: BLE001 - report to caller, don't crash
        return f"Agent backend unavailable: {exc}"
    if isinstance(response, dict) and response.get("payloads"):
        return "\n".join(item.get("text", "")
                         for item in response["payloads"])
    return json.dumps(response, ensure_ascii=False)[:4000]

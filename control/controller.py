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

"""Physician debate grounding (proposal.tmp Sections 4/8).

Debate answers must come from stored evidence (round report + audit
receipts), never from re-generated claims. This module builds the
grounded context; full transcripts reload on demand only.
"""

import json
from pathlib import Path

from control.paths import ROOT_DIR


def latest_report(record_index: int, root: Path = ROOT_DIR) -> Path | None:
    candidates = sorted(
        (root / "report").glob(f"round_{record_index}_*.md")
    )
    return candidates[-1] if candidates else None


def audit_for_record(record_index: int, root: Path = ROOT_DIR) -> list[dict]:
    path = root / "report" / "audit.jsonl"
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("record_index") == record_index:
            entries.append(entry)
    return entries


def build_debate_prompt(record_index: int, root: Path = ROOT_DIR) -> str:
    report = latest_report(record_index, root)
    if report is None:
        raise FileNotFoundError(
            f"No round report for record {record_index}. "
            "Run the round first: main.py --record-index "
            f"{record_index}"
        )
    audit_lines = "\n".join(
        json.dumps(e, ensure_ascii=False) for e in audit_for_record(record_index, root)
    )
    return (
        "You are the orchestrator discussing a completed screening round "
        "with the physician. Answer ONLY from the evidence below. If a "
        "question goes beyond it, say so and name the missing evidence.\n\n"
        f"--- ROUND REPORT ({report.name}) ---\n{report.read_text()}\n"
        f"--- AUDIT RECEIPTS ---\n{audit_lines or '(none)'}\n"
    )

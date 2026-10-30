"""Structured operational events for a screening round.

Events are deliberately small, JSONL-only records: stage, status, timing
and safe categorical/numeric dimensions. Raw prompts, steers and model
reasoning stay out of the event stream; they remain in the report/audit
where their audience is explicit. Event writes are best effort so a full
disk or unwritable report directory cannot break a round.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from control.paths import ROOT_DIR

SCHEMA_VERSION = 1


def new_run_id() -> str:
    return uuid.uuid4().hex


def event(run_id: str, stage: str, *, status: str = "ok", **fields) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "stage": stage,
        "status": status,
        **fields,
    }


def append_event(entry: dict, root: Path = ROOT_DIR) -> Path | None:
    path = Path(root) / "report" / "events.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path
    except OSError:
        return None

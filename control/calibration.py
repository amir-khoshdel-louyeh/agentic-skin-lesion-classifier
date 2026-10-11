"""Calibration registry: which diagnostic tools cleared the local gate.

`calibration/*.json` is written by `tools/calibrate.py` (local tools
only, no LLM/API). The diagnostic pool reads it so a tool that failed
calibration stays OUT of the pool even though its CLI still runs.

A missing or unreadable registry blocks nothing: a fresh clone without
calibration artifacts keeps working, and only measured failures gate.
"""

import json
from pathlib import Path

from control.paths import ROOT_DIR

CALIBRATION_DIRNAME = "calibration"
MIN_ACCURACY = 0.60
MAX_ECE = 0.10


def load_registry(root: Path = ROOT_DIR) -> dict[str, dict]:
    """tool name -> calibration payload (invalid files are ignored)."""
    directory = Path(root) / CALIBRATION_DIRNAME
    records: dict[str, dict] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("tool") or path.stem
        records[str(name)] = payload
    return records


def passed(record: dict) -> bool:
    gate = record.get("gate") or {}
    return bool(gate.get("passed"))


def blocked_tools(root: Path = ROOT_DIR) -> set[str]:
    """Names with a calibration record that did not pass the gate."""
    return {name for name, record in load_registry(root).items()
            if not passed(record)}

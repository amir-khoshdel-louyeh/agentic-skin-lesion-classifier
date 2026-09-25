"""Deterministic quality guard (Phase 10, no LLM).

Runs the OpenCV quality gate control-side and returns plain evidence.
A FAIL never diagnoses — the router treats it as heightened scrutiny,
not a stop: the reference demo image itself fails the blur bar, so a
hard stop would brick legitimate cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from control.paths import ROOT_DIR, resolve


def run_guard(image: str) -> dict:
    """Return {passed, metrics, flags, error}. Never raises."""
    cmd = [sys.executable, str(resolve("tools/quality_gate.py")),
           "--image", str(resolve(image))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120, cwd=str(ROOT_DIR))
    except Exception as exc:  # noqa: BLE001 - evidence, not crash
        return {"passed": False, "metrics": {}, "flags": ["tool_failed"],
                "error": str(exc)[:200]}
    if proc.returncode != 0:
        return {"passed": False, "metrics": {}, "flags": ["tool_failed"],
                "error": proc.stderr.strip()[-200:]}
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, TypeError):
        return {"passed": False, "metrics": {}, "flags": ["tool_failed"],
                "error": "no JSON output"}
    if payload.get("status") != "success":
        return {"passed": False, "metrics": {},
                "flags": ["tool_failed"],
                "error": str(payload.get("message"))[:200]}
    return {"passed": bool(payload.get("quality_pass")),
            "metrics": payload.get("metrics", {}),
            "flags": list(payload.get("uncertainty_flags") or []),
            "error": ""}

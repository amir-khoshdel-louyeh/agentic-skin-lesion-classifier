"""Deterministic quality guard (Phase 10, no LLM).

Runs the OpenCV quality gate control-side and returns plain evidence.
A FAIL never diagnoses — the router treats it as heightened scrutiny,
not a stop: the reference demo image itself fails the blur bar, so a
hard stop would brick legitimate cases.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from control.paths import ROOT_DIR, resolve
from control.tooljson import parse_tool_json


def run_guard(image: str) -> dict:
    """Return {passed, metrics, flags, error}. Never raises."""
    try:
        from control.vram import unload_ollama as _unload
        _unload()  # sequential GPU: LLM out before any tool loads
    except Exception:
        pass
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
        payload = parse_tool_json(proc.stdout)
    except (ValueError, TypeError) as exc:
        return {"passed": False, "metrics": {}, "flags": ["tool_failed"],
                "error": f"invalid tool JSON: {str(exc)[:200]}"}
    if payload.status != "success":
        return {"passed": False, "metrics": {},
                "flags": ["tool_failed"],
                "error": str(payload.message)[:200]}
    if payload.quality_pass is None:
        return {"passed": False, "metrics": {}, "flags": ["tool_failed"],
                "error": "quality gate omitted quality_pass"}
    return {"passed": payload.quality_pass,
            "metrics": payload.metrics or {},
            "flags": list(payload.uncertainty_flags),
            "error": ""}

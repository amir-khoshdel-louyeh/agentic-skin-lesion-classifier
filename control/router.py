"""Flag router (Phase 10, no LLM).

Pure evidence in, route out: {"path": "screen"|"full", "reasons": [...]}.
"screen" runs the triage worker only; "full" runs all three workers.
Any doubt defaults to full — the router may only save work, never risk.
"""

from __future__ import annotations

import json
import subprocess
import sys

from control.paths import ROOT_DIR, resolve

SCREEN_FLAGS = ("screen",)
FULL_FLAGS = ("full",)


def run_abcde(image: str) -> dict:
    """Return {risk_band, score, flags, error}. Never raises."""
    cmd = [sys.executable, str(resolve("tools/abcde_analyzer.py")),
           "--image", str(resolve(image))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=180, cwd=str(ROOT_DIR))
    except Exception as exc:  # noqa: BLE001 - evidence, not crash
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": str(exc)[:200]}
    if proc.returncode != 0:
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": proc.stderr.strip()[-200:]}
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, TypeError):
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": "no JSON output"}
    if payload.get("status") != "success":
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": str(payload.get("message"))[:200]}
    return {"risk_band": str(payload.get("risk_band", "")).lower(),
            "score": payload.get("abcde_score"),
            "flags": list(payload.get("uncertainty_flags") or []),
            "error": ""}


def route(guard: dict, abcde: dict) -> dict:
    """Decide the worker path from deterministic evidence."""
    reasons: list[str] = []
    if not guard.get("passed"):
        gflags = ",".join(guard.get("flags") or []) or "fail"
        reasons.append(f"quality gate: {gflags}")
    if "tool_failed" in guard.get("flags", []) + abcde.get("flags", []):
        reasons.append("evidence tool failed")
    band = abcde.get("risk_band", "")
    if band in ("moderate", "high"):
        reasons.append(f"abcde risk: {band} ({abcde.get('score')})")
    if not band:
        reasons.append("abcde risk unknown")
    if reasons:
        return {"path": "full", "reasons": reasons}
    return {"path": "screen",
            "reasons": [f"clean gate + abcde {band}"]}

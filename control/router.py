"""Tool wrappers for Quality Agent (plan.tmp T1 — decision logic RETIRED).

`route()` decision logic is DELETED per plan.tmp §6/T1: no deterministic
code may decide quality/route/tool/stop. What remains here are plain
tool callers (ABCDE + preprocess probe) for the Quality Agent (chain
step 1, §2.1) to invoke via its own LLM judgment. They return evidence,
never a path.
"""

from __future__ import annotations

import subprocess
import sys

from control.paths import ROOT_DIR, resolve
from control.tooljson import parse_tool_json


def run_abcde(image: str) -> dict:
    """Return {risk_band, score, flags, error}. Never raises."""
    try:
        from control.vram import unload_ollama as _unload
        _unload()  # sequential GPU: LLM out before any tool loads
    except Exception:
        pass
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
        payload = parse_tool_json(proc.stdout)
    except (ValueError, TypeError) as exc:
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": f"invalid tool JSON: {str(exc)[:200]}"}
    if payload.status != "success":
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": str(payload.message)[:200]}
    if not payload.risk_band:
        return {"risk_band": "", "score": None, "flags": ["tool_failed"],
                "error": "ABCDE tool omitted risk_band"}
    return {"risk_band": payload.risk_band.lower(),
            "score": payload.abcde_score,
            "flags": list(payload.uncertainty_flags),
            "error": ""}


def probe_preprocess(image: str, timeout: int = 300) -> dict:
    """Run preprocess to a temp file and report deltas. Never raises.

    Returns {hair_pixels, improved, error}: `improved` is True when the
    cleaned copy passes quality while the original did not.

    Retained on purpose (plan.tmp T1): the orchestrator does not call
    this — the Quality Agent runs `preprocess` through its own skill —
    but it stays as a plain, control-side evidence probe (tests /
    maintenance). It reports deltas only and never returns a route or
    decision, so it cannot become a decider.
    """
    import tempfile
    from pathlib import Path as _Path

    try:
        from control.vram import unload_ollama as _unload2
        _unload2()  # sequential GPU: LLM out before any tool loads
    except Exception:
        pass
    try:
        with tempfile.TemporaryDirectory(prefix="preprobe_") as tmp:
            out = str(_Path(tmp) / "cleaned.jpg")
            cmd = [sys.executable, str(resolve("tools/preprocess.py")),
                   "--image", str(resolve(image)),
                   "--output", out, "--steps", "hair,denoise"]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=str(ROOT_DIR))
            if proc.returncode != 0:
                return {"hair_pixels": 0, "improved": False,
                        "error": proc.stderr.strip()[-200:] or "exit!=0"}
            try:
                payload = parse_tool_json(proc.stdout)
            except (ValueError, TypeError) as exc:
                return {"hair_pixels": 0, "improved": False,
                        "error": f"invalid tool JSON: {str(exc)[:200]}"}
            if payload.status != "success":
                return {"hair_pixels": 0, "improved": False,
                        "error": str(payload.message)[:200]}
            if (payload.quality_pass_before is None
                    or payload.quality_pass_after is None
                    or payload.hair_pixels_inpainted is None):
                return {"hair_pixels": 0, "improved": False,
                        "error": "preprocess omitted quality/result fields"}
            return {"hair_pixels": payload.hair_pixels_inpainted,
                    "improved": (payload.quality_pass_before is False
                                 and payload.quality_pass_after is True),
                    "error": ""}
    except Exception as exc:  # noqa: BLE001 - evidence, not crash
        return {"hair_pixels": 0, "improved": False,
                "error": str(exc)[:200]}

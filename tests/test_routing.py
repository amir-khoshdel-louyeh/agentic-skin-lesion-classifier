"""Routing unit checks (stdlib only). Run: .venv/bin/python tests/test_routing.py
Exit non-zero on any failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.router import route  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


clean_guard = {"passed": True, "flags": [], "metrics": {}, "error": ""}
low_abcde = {"risk_band": "low", "score": 1.2, "flags": [], "error": ""}
r = route(clean_guard, low_abcde)
check("clean-goes-screen", r["path"] == "screen", str(r))

r = route({"passed": False, "flags": ["blurry"], "metrics": {},
           "error": ""}, low_abcde)
check("blurry-goes-full", r["path"] == "full", str(r))

r = route(clean_guard, {"risk_band": "moderate", "score": 4.76,
                        "flags": ["borderline"], "error": ""})
check("moderate-goes-full", r["path"] == "full", str(r))

r = route(clean_guard, {"risk_band": "", "score": None,
                        "flags": ["tool_failed"], "error": "x"})
check("tool-failure-goes-full", r["path"] == "full", str(r))

r = route({"passed": False, "flags": ["tool_failed"], "metrics": {},
           "error": "boom"},
          {"risk_band": "", "score": None, "flags": [], "error": ""})
check("guard-crash-goes-full", r["path"] == "full", str(r))

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

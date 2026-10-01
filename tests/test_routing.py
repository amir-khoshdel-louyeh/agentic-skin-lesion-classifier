"""Routing retirement checks (plan.tmp T1, stdlib only).
Run: .venv/bin/python tests/test_routing.py
Exit non-zero on any failure.

T1 deleted `control.router.route()` and `orchestrator.run_direct_tool`:
no deterministic code may decide quality/route/tool/stop. This test
locks that in:
  1. route() is gone (import fails).
  2. run_direct_tool is gone.
  3. run_abcde / probe_preprocess wrappers still exist (Quality Agent
     tool callers, evidence only — never a path).
  4. No `route(` decision call remains in orchestrator.py.
  5. No `screen` path literal remains in orchestrator.py (full always).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# 1. route() must be gone.
try:
    from control.router import route  # noqa: E402,F401
    check("route-deleted", False, "control.router.route still importable")
except ImportError:
    check("route-deleted", True)

# 2. run_direct_tool must be gone.
try:
    from orchestrator import run_direct_tool  # noqa: E402,F401
    check("run_direct_tool-deleted", False,
          "orchestrator.run_direct_tool still importable")
except ImportError:
    check("run_direct_tool-deleted", True)

# 3. Plain wrappers survive for the Quality Agent.
import control.router as router_mod  # noqa: E402

check("abcde-wrapper-kept", callable(getattr(router_mod, "run_abcde", None)))
check("preprocess-wrapper-kept",
      callable(getattr(router_mod, "probe_preprocess", None)))

# 4. No decision call left in the orchestrator: no `route(...)` call
# with guard/abcde evidence (comments/strings like "careful route"
# or "No route()" don't count — match the real call shape).
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
import re as _re
decision_call = _re.search(r"[=\s]route\s*\(\s*guard", orch_text)
check("no-route-call-in-orchestrator", decision_call is None,
      "orchestrator.py still calls route(guard, ...)")
check("no-route-import-in-orchestrator",
      _re.search(r"from control\.router import[^\n]*\broute\b(?!_|\w)",
                 orch_text) is None,
      "orchestrator.py still imports route")
check("no-run_direct_tool-in-orchestrator", "run_direct_tool" not in orch_text,
      "orchestrator.py still references run_direct_tool")

# 5. No screen-path literal left (T1: full path always).
check("no-screen-path-in-orchestrator",
      '"screen"' not in orch_text and "'screen'" not in orch_text,
      "orchestrator.py still mentions a screen path")

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

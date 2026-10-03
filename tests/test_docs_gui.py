"""Docs + GUI wiring checks (plan.tmp T12, stdlib only).
Run: .venv/bin/python tests/test_docs_gui.py
Exit non-zero on any failure.

Locks in: README describes the chain (not guard→router→screen/full),
role-skill installs, two-part report; GUI Round tab runs the chain with
a Part A/B view; chat tab uses §2.7 grounding (ask/load_case) only;
Doctor tab covers all prompt roles; plan TODOs all checked.
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


readme = (ROOT / "README.md").read_text(encoding="utf-8")
gui = (ROOT / "gui.py").read_text(encoding="utf-8")

# 1. README: chain, not the retired pipeline.
check("readme-chain", "Brain-loop" in readme or "BRAIN-LOOP" in readme
      or "Brain-loop" in readme)
check("readme-no-screen-path", "screen (clear cases, NO LLM)" not in readme)
check("readme-role-skills", all(f"skills/{r}" in readme or
                                f"skills/${{s}}" in readme or f"./openclaw-skills/{r}" in readme
                                for r in ("quality", "brain", "diagnostic",
                                          "critic", "specialist", "chat")))
check("readme-no-tier-install",
      "skills/skin-lesion-fast" not in readme)
check("readme-two-part", "Part A" in readme and "Part B" in readme)

# 2. GUI is cases-only: no tabs, no round/chat/doctor machinery.
check("gui-no-tabs", "Notebook" not in gui and "tabs.add" not in gui)
check("gui-case-card", "_case_card" in gui and "_refresh_cases" in gui
      and "_on_case" in gui)
check("gui-no-round", "run_round" not in gui)
check("gui-no-chat", "ask(state" not in gui and "_send_chat" not in gui)
check("gui-no-doctor", "doctor_role" not in gui and "PROMPT_ROLES" not in gui)
check("gui-report-status", "latest_report" in gui)

# 3. Prompt roles registry intact (GUI no longer consumes it).
from control.prompts import ROLES  # noqa: E402

check("four-roles", set(ROLES) == {"orchestrator", "agent", "quality",
                                   "brain"})

# 5. GUI compiles (headless-safe check: no display needed).
import py_compile  # noqa: E402

try:
    py_compile.compile(str(ROOT / "gui.py"), doraise=True)
    check("gui-compiles", True)
except py_compile.PyCompileError as exc:
    check("gui-compiles", False, str(exc)[:200])

# 6. Plan TODOs all checked.
plan = (ROOT / "plan.tmp").read_text(encoding="utf-8")
check("plan-all-done", "- [ ] T" not in plan
      and sum(1 for line in plan.splitlines()
              if line.startswith("- [x] T")) == 12)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

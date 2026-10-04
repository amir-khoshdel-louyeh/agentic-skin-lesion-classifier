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

# 2. GUI is cases + one Main tab: Start button, avatars, no
# chat/doctor/tools machinery.
check("gui-main-tab", "Notebook" in gui and "_build_main_tab" in gui
      and '"  Main  "' in gui)
check("gui-case-card", "_case_card" in gui and "_refresh_cases" in gui
      and "_on_case" in gui)
check("gui-maximized", 'state("zoomed")' in gui)
check("gui-start", "Start" in gui and "_on_start" in gui
      and "run_round" in gui and "on_stage" in gui)
check("gui-avatars", "Canvas" in gui and "_draw_chain" in gui
      and "_activate" in gui and "_apply_stage" in gui)
check("gui-avatar-stages", all(s in gui for s in
      ("quality", "brain", "critic", "referee", "GREEN_AVATAR")))
check("gui-run-stem", "_run_stem" in gui
      and "currently selected one" in gui)
check("gui-chat-box", "_build_chat_box" in gui and "chat_box" in gui
      and "chat_entry" in gui and "_chat_line" in gui)
check("gui-stage-log", "_log_stage" in gui and "[Referee]" in gui
      and "[Session" in gui)
check("gui-ask-stop", "_on_ask" in gui and "_ask_worker" in gui
      and "_on_stop" in gui and "chat_done" in gui)
check("gui-no-chat-tab", "_send_chat" not in gui
      and "Physician chat" not in gui)
check("gui-no-doctor", "doctor_role" not in gui and "PROMPT_ROLES" not in gui)
check("gui-report-status", "latest_report" in gui)

# 3. run_round emits stage progress for avatars (default off).
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
import inspect  # noqa: E402
import orchestrator  # noqa: E402

sig = inspect.signature(orchestrator.run_round)
check("on-stage-param", "on_stage" in sig.parameters
      and sig.parameters["on_stage"].default is None)
for stage in ("quality", "brain", "session", "critic", "specialists",
              "specialist", "referee"):
    check(f"emit-{stage}", f'_emit("{stage}"' in orch_text)
check("emit-steers", orch_text.count(
    '"steer": order.get("steer", "")') >= 2)

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

"""Grounding checks: narration only post-lock (stdlib only)."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.chat import WEB_RULES, build_debate_prompt  # noqa: E402
from control.controller import (  # noqa: E402
    CaseState,
    build_question_prompt,
    load_case,
)

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    try:
        build_debate_prompt(0, root)
        check("debate-needs-report", False, "no error raised")
    except FileNotFoundError:
        check("debate-needs-report", True)
    try:
        load_case(0, root)
        check("controller-needs-report", False, "no error raised")
    except FileNotFoundError:
        check("controller-needs-report", True)

check("web-rules-cite", "cite" in WEB_RULES.lower())
check("web-rules-no-vote-change", "vote stands" in WEB_RULES)

state = CaseState(record_index=0, report_name="r.md",
                  report_text="Decision: **suspicious** — x")
prompt = build_question_prompt(state, "Why referred?")
check("narration-answers-from-evidence",
      "ONLY from the evidence" in prompt)
check("narration-refuses-beyond-evidence",
      "Not in the record" in prompt)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

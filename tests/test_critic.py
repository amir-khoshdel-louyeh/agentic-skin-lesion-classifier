"""Critic checks (plan.tmp T6, stdlib only).
Run: .venv/bin/python tests/test_critic.py
Exit non-zero on any failure.

Locks in §2.4: EXACTLY ONCE per round, post-loop only (never in-loop,
never per-session); read-only receipts + Quality flags; object-with-cites
or silence; invalid output = silence; sustained objection forces
suspicious-refer.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.critic import (  # noqa: E402
    build_critic_prompt,
    parse_objection,
    run_critic,
)
from control.decide import decide  # noqa: E402
from control.schemas import VerdictEnvelope  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def verdict(cls="Melanocytic nevi", conf=0.85, flags=()):
    return VerdictEnvelope(
        ran=True, command="python tools/ham10000_cnn.py --image i.jpg",
        exit_code=0, predicted_class=cls, confidence=conf,
        reasoning="r", uncertainty_flags=list(flags))


def obj(objection=True, reasons=None, cites=None):
    return json.dumps({"objection": objection,
                       "reasons": reasons if reasons is not None else ["low"],
                       "cites": cites if cites is not None else ["Verdict 1"]})


# 1. Object-with-cites sustains; everything else is silence.
sustained, _ = parse_objection(obj(), 2)
check("sustains-with-cites", sustained is True)
for name, raw in (
        ("false-is-silence", obj(objection=False)),
        ("missing-reasons-silence", obj(reasons=[])),
        ("missing-cites-silence", obj(cites=[])),
        ("out-of-range-cite-silence", obj(cites=["Verdict 9"])),
        ("garbage-silence", "no json here"),
        ("empty-silence", ""),
        ("non-object-silence", "[1, 2]")):
    sustained, _ = parse_objection(raw, 2)
    check(name, sustained is False, raw[:60])

# 2. Backend failure = silence, never an exception.
import skin_agent  # noqa: E402

_real = skin_agent.run_openclaw_cli
skin_agent.run_openclaw_cli = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("no backend"))
try:
    sustained, reasons = run_critic([verdict()], [], agent_id="main")
    check("backend-failure-silence", sustained is False, str(reasons))
except Exception as exc:  # noqa: BLE001
    check("backend-failure-silence", False, f"raised: {exc}")
finally:
    skin_agent.run_openclaw_cli = _real

# 3. Prompt: receipts + Quality flags, asymmetric role, no tools.
prompt = build_critic_prompt([verdict("Melanoma", 0.62, ["borderline"])],
                             ["blurry"], "Quality: flagged — blur")
check("prompt-has-receipts", "Melanoma" in prompt and "0.62" in prompt)
check("prompt-has-quality", "Quality: flagged" in prompt)
check("prompt-asymmetric", "may NOT" in prompt and "confirm" in prompt)
check("prompt-cites-required", "MUST cite" in prompt)

# 4. Sustained objection forces suspicious-refer; silence changes nothing.
d = decide([verdict()], critic_objection=True)
check("sustained-forces-suspicious", d["decision"] == "suspicious",
      str(d))
d = decide([verdict()], critic_objection=False)
check("silence-changes-nothing", d["decision"] == "benign", str(d))

# 5. Skill file exists, read-only, object-only.
skill = ROOT / "openclaw-skills" / "critic" / "SKILL.md"
check("skill-exists", skill.exists())
if skill.exists():
    text = skill.read_text(encoding="utf-8")
    check("skill-read-only", "NO tools" in text)
    check("skill-once-post-loop", "EXACTLY ONCE" in text
          and "AFTER" in text)
    check("skill-object-only", "ONLY for rejection" in text
          or "ONLY job is to object" in text)

# 6. Wiring: exactly one call site, post-loop (after workers_s timing),
# never inside the Brain-loop; Quality text fed in.
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("critic-called-once", orch_text.count("run_critic(") == 1,
      f"{orch_text.count('run_critic(')} call sites")
check("critic-post-loop", orch_text.index('timings["workers_s"]')
      < orch_text.index("run_critic("))
check("critic-gets-quality", "quality_text=quality_text" in orch_text)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

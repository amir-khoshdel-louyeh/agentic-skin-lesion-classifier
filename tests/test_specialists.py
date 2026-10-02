"""Specialist checks (plan.tmp T7, stdlib only).
Run: .venv/bin/python tests/test_specialists.py
Exit non-zero on any failure.

Locks in §2.5: code-counted trigger (≥2 distinct countable classes,
top-2 by re-executed confidence), Brain-written angle per disease,
off-shortlist ⇒ no-evidence, persistent disagreement ⇒ named dilemma
(never averaged).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.brain import (  # noqa: E402
    parse_specialist_briefs,
    run_specialist_briefs,
)
from control.schemas import VerdictEnvelope  # noqa: E402
from control.specialists import (  # noqa: E402
    advocate_brief,
    shortlist,
    specialist_brief,
)

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def env(cls, conf, ran=True):
    return VerdictEnvelope(
        ran=ran, command="python tools/ham10000_cnn.py --image i.jpg",
        exit_code=0 if ran else 1, predicted_class=cls if ran else None,
        confidence=conf if ran else None, reasoning="r",
        uncertainty_flags=[])


# 1. Trigger is pure counting: top-2 by confidence, None otherwise.
got = shortlist([env("Melanoma", 0.9), env("Melanocytic nevi", 0.8),
                 env("Melanoma", 0.7)])
check("top-2-by-confidence", got == ["Melanoma", "Melanocytic nevi"],
      str(got))
check("single-class-skips", shortlist(
    [env("Melanoma", 0.9), env("Melanoma", 0.7)]) is None)
check("no-countable-skips", shortlist(
    [env("Melanoma", 0.9, ran=False)]) is None)
check("three-way-caps-at-2", len(shortlist(
    [env("Melanoma", 0.9), env("Basal cell carcinoma", 0.8),
     env("Vascular lesions", 0.7)])) == 2)

# 2. Brain briefs: exact coverage wins; anything else falls back.
diseases = ["Melanoma", "Melanocytic nevi"]


def briefs(payload):
    return json.dumps(payload)


valid = briefs({"briefs": [{"disease": "Melanoma", "angle": "Stress asymmetry."},
                            {"disease": "Melanocytic nevi",
                             "angle": "Stress symmetry + history."}]})
out = parse_specialist_briefs(valid, diseases)
check("valid-briefs", out is not None
      and set(out) == set(diseases)
      and "asymmetry" in out["Melanoma"], str(out))
for name, raw in (
        ("garbage-fallback", "no json"),
        ("missing-disease-fallback", briefs({"briefs": [
            {"disease": "Melanoma", "angle": "x"}]})),
        ("extra-disease-fallback", briefs({"briefs": [
            {"disease": "Melanoma", "angle": "x"},
            {"disease": "Melanocytic nevi", "angle": "y"},
            {"disease": "Other", "angle": "z"}]})),
        ("empty-angle-fallback", briefs({"briefs": [
            {"disease": "Melanoma", "angle": "  "},
            {"disease": "Melanocytic nevi", "angle": "y"}]})),
        ("duplicate-fallback", briefs({"briefs": [
            {"disease": "Melanoma", "angle": "x"},
            {"disease": "Melanoma", "angle": "y"}]}))):
    check(name, parse_specialist_briefs(raw, diseases) is None, raw[:60])

# 3. Backend failure / invalid output ⇒ static fallback, never raises.
import skin_agent  # noqa: E402

_real = skin_agent.run_openclaw_cli
skin_agent.run_openclaw_cli = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("no backend"))
try:
    briefs_out, source = run_specialist_briefs(
        diseases, [], "Quality: pass", agent_id="main")
    check("backend-fallback", source == "fallback"
          and set(briefs_out) == set(diseases), f"{source} {briefs_out}")
except Exception as exc:  # noqa: BLE001
    check("backend-fallback", False, f"raised: {exc}")
finally:
    skin_agent.run_openclaw_cli = _real

# 4. Advocate brief carries the Brain angle on top of the static role.
text = advocate_brief("Melanoma", diseases, "Stress asymmetry.")
check("advocate-has-angle", "Stress asymmetry." in text)
check("advocate-keeps-role", "advocate ONLY for Melanoma" in text)
check("fallback-template-kept", "advocate ONLY for Melanoma" in
      specialist_brief("Melanoma", diseases))

# 5. Skill file exists, advocate-only.
skill = ROOT / "openclaw-skills" / "specialist" / "SKILL.md"
check("skill-exists", skill.exists())
if skill.exists():
    text = skill.read_text(encoding="utf-8")
    check("skill-advocate-only", "advocate" in text.lower()
          and "FOR" in text)
    check("skill-no-averaging", "never" in text.lower()
          and "averag" in text.lower())

# 6. Wiring: Brain briefs drive advocates; off-shortlist + dilemma stay.
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("orchestrator-brain-briefs", "run_specialist_briefs(" in orch_text)
check("orchestrator-advocate-brief", "advocate_brief(" in orch_text)
check("orchestrator-off-shortlist", "off-shortlist" in orch_text)
check("orchestrator-dilemma", "specialist dilemma" in orch_text)
check("orchestrator-brief-audit", '"brief_source"' in orch_text)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

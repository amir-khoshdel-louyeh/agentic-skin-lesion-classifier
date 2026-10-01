"""Brain supervisor checks (plan.tmp T3, stdlib only).
Run: .venv/bin/python tests/test_brain.py
Exit non-zero on any failure.

Locks in §2.2: text-only orders (no images, no tools), SHORT steer per
session, suggested_subset as hint only, Brain-decided 1..3 sessions with
early stop, fail-safe stop on invalid JSON, every order + why audited.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.brain import (  # noqa: E402
    MAX_SESSIONS,
    build_brain_prompt,
    diagnostic_pool,
    parse_brain_order,
    receipt_summary,
    run_brain_step,
)
from control.manifest import load_manifest  # noqa: E402
from control.prompts import load_role  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


POOL = ["ham10000-cnn", "multimodal-fusion", "ensemble-high"]


def order(steer="Look closely at the border.", subset=None, stop=False,
          why="first look"):
    return json.dumps({"steer": steer,
                       "suggested_subset": subset if subset is not None
                       else ["ham10000-cnn"],
                       "stop": stop, "why": why})


# 1. Valid orders parse through; unknown hint names drop (hint-only).
out = parse_brain_order(order(), POOL)
check("parse-valid", out["stop"] is False
      and out["suggested_subset"] == ["ham10000-cnn"]
      and out["why"] == "first look", str(out))
out = parse_brain_order(order(subset=["ham10000-cnn", "nope-tool"]), POOL)
check("unknown-hint-dropped", out["stop"] is False
      and out["suggested_subset"] == ["ham10000-cnn"], str(out))
out = parse_brain_order(order(steer="", stop=True, why="enough evidence"),
                        POOL)
check("stop-order-valid", out["stop"] is True, str(out))

# 2. Fail-safe is always stop=True (never an unbounded loop).
check("invalid-json-stops",
      parse_brain_order("garbage", POOL)["stop"] is True)
check("missing-why-stops",
      parse_brain_order(order(why=""), POOL)["stop"] is True)
check("nonbool-stop-stops",
      parse_brain_order(order().replace('"stop": false', '"stop": "no"'),
                        POOL)["stop"] is True)
check("continue-without-steer-stops",
      parse_brain_order(order(steer="  ", stop=False), POOL)["stop"] is True)
check("bad-subset-stops",
      parse_brain_order(order(subset="ham10000-cnn"), POOL)["stop"] is True)
for garbage in ("", "{}", "[]", '{"stop": null}', "{unclosed"):
    try:
        ok = parse_brain_order(garbage, POOL)["stop"] is True
    except Exception:  # noqa: BLE001 - parser must never raise
        ok = False
    check(f"never-raises:{garbage[:12]!r}", ok)
check("max-sessions-cap", MAX_SESSIONS == 3)

# 3. Backend failure becomes a stop order, never an exception.
import skin_agent  # noqa: E402

_real = skin_agent.run_openclaw_cli


def _boom(*args, **kwargs):
    raise RuntimeError("no backend")


skin_agent.run_openclaw_cli = _boom
try:
    out = run_brain_step({"image_path": "x.jpg", "metadata": {}},
                         "Quality: pass", [], POOL, agent_id="main")
    check("backend-failure-stops", out["stop"] is True, str(out))
except Exception as exc:  # noqa: BLE001
    check("backend-failure-stops", False, f"raised: {exc}")
finally:
    skin_agent.run_openclaw_cli = _real

# 4. Role files: locked layer carries the hard boundary + hint rule.
role_text = load_role("brain")
check("role-loads", "steer" in role_text.lower())
check("role-text-only", "never see images" in role_text)
check("role-hint-only", "hint only" in role_text)

# 5. Prompt is text-only: no image path may leak to the Brain.
manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
pool = diagnostic_pool(manifest)
check("pool-excludes-vlm", "drdiag-vlm" not in pool
      and "vlm-ollama" not in pool, str(pool))
check("pool-has-diagnostics", "ham10000-cnn" in pool
      and "multimodal-fusion" in pool, str(pool))
record = {"image_path": "dataset/secret_image.jpg",
          "metadata": {"age": 45}}
prompt = build_brain_prompt(record, "Quality: flagged — blur", [], role_text,
                            pool)
check("prompt-hides-image", "secret_image.jpg" not in prompt)
check("prompt-has-quality", "Quality: flagged" in prompt)
check("prompt-has-metadata", "45" in prompt)
hist = [{"n": 1, "class": "Melanoma", "confidence": 0.62,
         "flags": "borderline", "tools": "Triage",
         "reasoning": "low confidence borderline"}]
prompt2 = build_brain_prompt(record, "Quality: pass", hist, role_text, pool)
check("prompt-feeds-receipts", "Melanoma" in prompt2
      and "0.62" in prompt2 and "secret_image.jpg" not in prompt2)

# 6. Receipt summaries carry re-executed numbers, not prose.
from control.schemas import VerdictEnvelope  # noqa: E402

v = VerdictEnvelope(ran=True, command="python tools/x.py --image i.jpg",
                    exit_code=0, predicted_class="Melanoma", confidence=0.62,
                    reasoning="r", uncertainty_flags=["borderline"])
s = receipt_summary(v, 1, tools="Triage")
check("receipt-summary", s["class"] == "Melanoma"
      and s["confidence"] == 0.62 and s["n"] == 1, str(s))

# 7. Skill file exists, text-only (no diagnostic commands).
skill = ROOT / "openclaw-skills" / "brain" / "SKILL.md"
check("skill-exists", skill.exists())
if skill.exists():
    text = skill.read_text(encoding="utf-8")
    check("skill-no-diagnostic-tools", "ham10000_cnn" not in text
          and "multimodal_fusion" not in text)
    check("skill-hint-only", "hint only" in text)

# 8. Chain step 2 wiring: Brain drives sessions, orders audited.
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("orchestrator-runs-brain", "run_brain_step(" in orch_text)
check("orchestrator-no-fixed-loop", "for extra in AGENT_ROLES" not in
      orch_text, "fixed AGENT_ROLES loop still drives sessions")
check("orchestrator-max-cap", "MAX_SESSIONS" in orch_text)
check("orchestrator-steers-audit", '"steers": steers' in orch_text
      or "'steers': steers" in orch_text or '"steers"' in orch_text)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

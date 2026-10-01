"""Quality Agent checks (plan.tmp T2, stdlib only).
Run: .venv/bin/python tests/test_quality.py
Exit non-zero on any failure.

Locks in §2.1: pass|cleaned|flagged schema with cites, never-diagnose /
never-stop rules, fail-safe to flagged (never an abort), quality skill
toolkit, and chain-step-1 wiring in the orchestrator.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.manifest import load_manifest, render_subset  # noqa: E402
from control.prompts import load_role  # noqa: E402
from control.quality import (  # noqa: E402
    QUALITY_TOOLKIT,
    build_quality_prompt,
    parse_quality,
    quality_line,
    run_quality,
)

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def q(verdict, reasons=None, cites=None, cleaned_image=None):
    payload = {"verdict": verdict,
               "reasons": reasons if reasons is not None else ["r"],
               "cites": cites if cites is not None else ["quality_gate: x"],
               "cleaned_image": cleaned_image}
    return json.dumps(payload)


# 1. Valid verdicts parse through.
for verdict in ("pass", "flagged"):
    out = parse_quality(q(verdict))
    check(f"parse-{verdict}", out["verdict"] == verdict, str(out))
out = parse_quality(q("cleaned", cleaned_image="/tmp/cleaned.jpg"))
check("parse-cleaned-with-path", out["verdict"] == "cleaned"
      and out["cleaned_image"] == "/tmp/cleaned.jpg", str(out))

# 2. Fail-safe is always flagged (never a stop, never raises).
check("invalid-json-flagged",
      parse_quality("not json at all")["verdict"] == "flagged")
check("bad-verdict-flagged",
      parse_quality(q("excellent"))["verdict"] == "flagged")
check("missing-reasons-flagged",
      parse_quality(q("pass", reasons=[]))["verdict"] == "flagged")
check("missing-cites-flagged",
      parse_quality(q("pass", cites=[]))["verdict"] == "flagged")
check("cleaned-without-path-flagged",
      parse_quality(q("cleaned"))["verdict"] == "flagged")
for garbage in ("", "{}", "[]", '{"verdict": null}', "None", "{unclosed"):
    try:
        out = parse_quality(garbage)
        ok = out["verdict"] == "flagged"
    except Exception:  # noqa: BLE001 - parser must never raise
        ok = False
    check(f"never-raises:{garbage[:12]!r}", ok)

# 3. Backend failure becomes flagged, never an exception.
import skin_agent  # noqa: E402

_real = skin_agent.run_openclaw_cli


def _boom(*args, **kwargs):
    raise RuntimeError("no backend")


skin_agent.run_openclaw_cli = _boom
try:
    out = run_quality({"image_path": "x.jpg", "metadata": {}},
                      "toolkit", agent_id="main")
    check("backend-failure-flagged", out["verdict"] == "flagged", str(out))
except Exception as exc:  # noqa: BLE001
    check("backend-failure-flagged", False, f"raised: {exc}")
finally:
    skin_agent.run_openclaw_cli = _real

# 4. Role files: locked layer carries the rules.
role_text = load_role("quality")
check("role-loads", "quality_gate" in role_text)
check("role-never-diagnoses", "never diagnoses" in role_text)
check("role-never-stops", "never stops the round" in role_text)

# 5. Prompt shows case + toolkit-only subset + schema.
manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
toolkit_text = render_subset(manifest, QUALITY_TOOLKIT)
for name in QUALITY_TOOLKIT:
    check(f"toolkit-has-{name}", name in toolkit_text, toolkit_text[:200])
check("toolkit-hides-diagnostics", "ham10000-cnn" not in toolkit_text,
      toolkit_text[:300])
prompt = build_quality_prompt({"image_path": "img.jpg", "metadata": {}},
                              role_text, toolkit_text)
check("prompt-has-schema", "pass|cleaned|flagged" in prompt)
check("prompt-has-image", "img.jpg" in prompt)

# 6. Skill file exists with the toolkit.
skill = ROOT / "openclaw-skills" / "quality" / "SKILL.md"
check("skill-exists", skill.exists())
if skill.exists():
    text = skill.read_text(encoding="utf-8")
    check("skill-quality-gate", "quality_gate" in text)
    check("skill-preprocess", "preprocess" in text)
    check("skill-abcde", "abcde_analyzer" in text)
    check("skill-never-diagnoses", "never diagnoses" in text)

# 7. Chain step 1 wiring: orchestrator spawns Quality first, audits it.
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("orchestrator-runs-quality", "run_quality(" in orch_text)
check("orchestrator-quality-audit", '"quality": quality' in orch_text
      or "'quality': quality" in orch_text or '"quality":' in orch_text)
check("orchestrator-quality-timing", '"quality_s"' in orch_text
      or "'quality_s'" in orch_text)
check("quality-line-renders", quality_line(
    {"verdict": "flagged", "reasons": ["blur"],
     "cites": ["c"], "cleaned_image": None}).startswith("Quality: flagged"))

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

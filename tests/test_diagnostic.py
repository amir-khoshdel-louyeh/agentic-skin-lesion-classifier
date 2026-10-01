"""Diagnostic-session checks (plan.tmp T4, stdlib only).
Run: .venv/bin/python tests/test_diagnostic.py
Exit non-zero on any failure.

Locks in §2.3: no `use ONLY` role scripts, every session gets case +
Quality + steer + FULL diagnostic pool (VLM excluded, support tools
hidden), free tool choice, one repair retry with double failure as
ran=False no-evidence, per-session audit log.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.runner import build_agent_prompt, run_agent  # noqa: E402
from control.brain import diagnostic_pool, session_tools  # noqa: E402
from control.manifest import load_manifest, render_subset  # noqa: E402
from control.prompts import load_role  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
pool = diagnostic_pool(manifest)
diag_text = render_subset(manifest, tuple(pool))

# 1. FULL pool: diagnostics in, VLM + support tools out.
for name in ("ham10000-cnn", "multimodal-fusion", "ensemble-high",
             "dermai-b0"):
    check(f"pool-has-{name}", name in diag_text, diag_text[:300])
check("pool-excludes-vlm", "drdiag-vlm" not in diag_text
      and "vlm-ollama" not in diag_text, diag_text[:300])
for name in ("quality-gate", "preprocess", "abcde-analyzer"):
    check(f"pool-hides-{name}", name not in diag_text, diag_text[:300])

# 2. No `use ONLY <role>` scripts anywhere in the session path. (A pool
# boundary — "ONLY tools listed in your pool, never invent commands" —
# stays: it bans hallucinated tools without assigning role scripts.)
role_text = load_role("agent")
check("role-allows-free-choice", "matching your role" not in role_text
      and "Choose freely" in role_text, role_text)
check("role-keeps-honesty", "Never fabricate tool output" in role_text)
check("role-keeps-borderline", "do not rely on me" in role_text)
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("no-agent-roles", "AGENT_ROLES" not in orch_text,
      "fixed AGENT_ROLES scripts still exist")
check("sessions-get-full-pool", "diag_manifest_text" in orch_text)

# 3. Every session prompt carries case + Quality + steer + pool.
record = {"image_path": "img.jpg", "metadata": {"age": 50}}
prompt = build_agent_prompt(record, diag_text, quality_text="Quality: pass",
                            steer="Check the border.", hint=["ham10000-cnn"])
check("prompt-has-image", "img.jpg" in prompt)
check("prompt-has-quality", "Quality: pass" in prompt)
check("prompt-has-steer", "Check the border." in prompt)
check("prompt-hint-is-hint", "HINT only" in prompt)
check("prompt-has-pool", "ham10000-cnn" in prompt
      and "multimodal-fusion" in prompt)
check("prompt-hides-support", "quality_gate" not in prompt
      and "abcde_analyzer" not in prompt)
check("prompt-isolates-sessions", "invisible to you" in prompt
      or "THIS session only" in prompt)

# 4. Free choice flows through: session output parses; tools derived
# control-side from the verified command, never from prose.
# NOTE: agents/runner.py binds run_openclaw_cli at import time, so the
# patch target is agents.runner — patching skin_agent would call the
# REAL backend here.
import agents.runner as runner_mod  # noqa: E402

_real = runner_mod.run_openclaw_cli
seen = {}


def _ok(prompt, agent_id="main", show_command=True):
    seen["prompt"] = prompt
    return {"payloads": [{"text": json.dumps(
        {"ran": True,
         "command": "python tools/ham10000_cnn.py --image img.jpg",
         "exit_code": 0, "predicted_class": "Melanoma",
         "confidence": 0.8, "reasoning": "t",
         "uncertainty_flags": []})}]}
runner_mod.run_openclaw_cli = _ok
try:
    out = run_agent(record, diag_text, quality_text="Quality: pass",
                    steer="Check.", hint=["ham10000-cnn"])
    check("session-parses", out.ran and out.predicted_class == "Melanoma",
          str(out))
    check("session-got-quality-steer", "Quality: pass" in seen.get(
        "prompt", "") and "Check." in seen.get("prompt", ""))
finally:
    runner_mod.run_openclaw_cli = _real


def _garbage(*args, **kwargs):
    return {"payloads": [{"text": "definitely not json"}]}
runner_mod.run_openclaw_cli = _garbage
try:
    out = run_agent(record, diag_text)
    check("double-failure-no-evidence", out.ran is False, str(out))
except Exception as exc:  # noqa: BLE001
    check("double-failure-no-evidence", False, f"raised: {exc}")
finally:
    runner_mod.run_openclaw_cli = _real

# 5. session_tools derives the run tool control-side.
check("session-tools-extracts",
      session_tools("python tools/ensemble_high.py --image i.jpg "
                    "--metadata {}") == "ensemble_high")
check("session-tools-unknown",
      session_tools("(agent output rejected)") == "unknown")

# 6. Orchestrator logs per-session steer/tools/receipts.
check("orchestrator-sessions-audit", '"sessions": sessions_log' in orch_text
      or "'sessions': sessions_log" in orch_text
      or '"sessions"' in orch_text)
check("orchestrator-session-tools", "session_tools(" in orch_text)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

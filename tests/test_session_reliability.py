"""Session reliability checks (stdlib only, no LLM/tools needed).
Run: .venv/bin/python tests/test_session_reliability.py
Exit non-zero on any failure.

The core guarantee: every sub-agent can execute a tool and return an
answer without fabrication. Command-form errors are eliminated by
construction (pre-rendered EXACT lines — nothing to retype), residual
receipt rejections get exactly one targeted repair, genuine ran=false
passes through untouched, and re-execution stays the trust anchor.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agents.runner as runner_mod  # noqa: E402
from agents.runner import (  # noqa: E402
    build_agent_prompt,
    build_receipt_repair_prompt,
    run_session,
)
from control.brain import diagnostic_pool  # noqa: E402
from control.manifest import (  # noqa: E402
    load_manifest,
    render_subset_for_case,
)
from control.schemas import VerdictEnvelope  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
pool = diagnostic_pool(manifest)
record = {"image_path": "dataset/ISIC2019_test/ISIC_0000002.jpg",
          "metadata": {"patient_id": "A001", "age": 60, "sex": "female"}}
case_text = render_subset_for_case(manifest, tuple(pool), record)

# 1. Pre-rendered lines: complete, runnable, nothing to fill in.
check("lines-have-real-image",
      "dataset/ISIC2019_test/ISIC_0000002.jpg" in case_text)
check("lines-have-real-metadata",
      '"patient_id": "A001"' in case_text or '"patient_id":"A001"' in
      case_text.replace(" ", ""))
check("lines-need-metadata-only-where-required",
      "--metadata" in [line for line in case_text.splitlines()
                       if "multimodal-fusion" in line][0])
cmd_lines = [line for line in case_text.splitlines()
             if line.startswith("- ")]
check("lines-no-placeholders", all("<path" not in line
      and "<json" not in line for line in cmd_lines)
      and len(cmd_lines) == len(pool))
check("lines-python-prefix", all(
    line.split("`")[1].startswith("python tools/")
    for line in case_text.splitlines() if line.startswith("- ")))
try:
    render_subset_for_case(manifest, ("nope-tool",), record)
    check("unknown-tool-fails-fast", False, "no KeyError raised")
except KeyError:
    check("unknown-tool-fails-fast", True)

# 2. Prompt: exact-lines choice + prerequisite ban.
prompt = build_agent_prompt(record, case_text, quality_text="Quality: pass",
                            steer="Check.", hint=["ham10000-cnn"])
check("prompt-exact-lines", "EXACTLY as written" in prompt
      and "copy-paste" in prompt)
check("prompt-no-prerequisite", "NO" in prompt
      and "cleaned-image prerequisite" in prompt
      and "Never refuse for a missing cleaned image" in prompt)

# 3. run_session: claim → verify → one targeted repair → verify.
calls = []


def _envelope(command, ran=True, cls="Melanoma", conf=0.8):
    return {"ran": ran, "command": command,
            "exit_code": 0 if ran else 1,
            "predicted_class": cls if ran else None,
            "confidence": conf if ran else None, "reasoning": "r",
            "uncertainty_flags": []}


GOOD = ("python tools/ham10000_cnn.py --image "
        "dataset/ISIC2019_test/ISIC_0000002.jpg")
BAD = "skin-lesion-mid --image dataset/ISIC2019_test/ISIC_0000002.jpg"


def _verify(claimed):
    if claimed.command == GOOD:
        return VerdictEnvelope(
            ran=True, command=GOOD, exit_code=0,
            predicted_class="Melanoma", confidence=0.7,
            reasoning="re-executed", uncertainty_flags=[])
    return VerdictEnvelope(ran=False, command=claimed.command,
                           reasoning="Receipt rejected: command not in "
                                     "verifiable form",
                           uncertainty_flags=["tool_failed"])


def _cli_factory(scripts):
    def _cli(prompt, agent_id="main", show_command=True):
        calls.append(prompt)
        return {"payloads": [{"text": json.dumps(scripts[len(calls) - 1])}]}
    return _cli


_real = runner_mod.run_openclaw_cli

# 3a. Bad command → one repair → good command verifies.
calls.clear()
runner_mod.run_openclaw_cli = _cli_factory(
    [_envelope(BAD), _envelope(GOOD)])
try:
    out = run_session(record, case_text, _verify, quality_text="Q",
                      steer="S")
    check("repair-converts-rejection", out.ran is True
          and out.predicted_class == "Melanoma"
          and len(calls) == 2, f"{out} calls={len(calls)}")
    check("repair-prompt-names-defect", BAD in calls[1]
          and "REJECTED" in calls[1] and "EXACT lines" in calls[1])
except Exception as exc:  # noqa: BLE001
    check("repair-converts-rejection", False, f"raised: {exc}")
    check("repair-prompt-names-defect", False, f"raised: {exc}")
finally:
    runner_mod.run_openclaw_cli = _real

# 3b. Genuine ran=false passes through with NO retry.
calls.clear()
runner_mod.run_openclaw_cli = _cli_factory([_envelope("x", ran=False)])
try:
    out = run_session(record, case_text, _verify)
    check("genuine-false-untouched", out.ran is False
          and len(calls) == 1, f"{out} calls={len(calls)}")
except Exception as exc:  # noqa: BLE001
    check("genuine-false-untouched", False, f"raised: {exc}")
finally:
    runner_mod.run_openclaw_cli = _real

# 3c. Repair also bad → first rejection stands, exactly 2 calls.
calls.clear()
runner_mod.run_openclaw_cli = _cli_factory(
    [_envelope(BAD), _envelope("also bad")])
try:
    out = run_session(record, case_text, _verify)
    check("double-rejection-stands", out.ran is False
          and len(calls) == 2, f"{out} calls={len(calls)}")
except Exception as exc:  # noqa: BLE001
    check("double-rejection-stands", False, f"raised: {exc}")
finally:
    runner_mod.run_openclaw_cli = _real

# 3d. Verifier crash becomes no-evidence, never raises.


def _boom_verify(claimed):
    raise RuntimeError("verifier down")


runner_mod.run_openclaw_cli = _cli_factory([_envelope(GOOD)])
try:
    out = run_session(record, case_text, _boom_verify)
    check("verify-crash-safe", out.ran is False, str(out))
except Exception as exc:  # noqa: BLE001
    check("verify-crash-safe", False, f"raised: {exc}")
finally:
    runner_mod.run_openclaw_cli = _real

# 4. Repair prompt carries the defect + exact lines + schema.
repair = build_receipt_repair_prompt(BAD, "Receipt rejected: bad form",
                                     case_text, "Q", "S")
check("repair-prompt-complete", BAD in repair and "bad form" in repair
      and "EXACT lines" in repair and "Reply ONCE" in repair
      and "JSON object" in repair)

# 5. Skill documents exact-lines + ban.
skill = (ROOT / "openclaw-skills" / "diagnostic" / "SKILL.md"
         ).read_text(encoding="utf-8")
check("skill-exact-lines", "copy-paste" in skill
      and "EXACT pre-filled" in skill)
check("skill-no-prerequisite", "NO" in skill
      and "cleaned-image prerequisite" in skill)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

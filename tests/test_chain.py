"""Chain acceptance checks (plan.tmp T11 + §8, stdlib only).
Run: .venv/bin/python tests/test_chain.py
Exit non-zero on any failure.

Locks in the acceptance list: chain order Quality → Brain-loop →
Critic → Specialist → Referee; critic exactly once post-loop;
autonomy (same image MAY differ — code cannot pin tools or steers);
per-session audit (steer+why, tools, re-executed numbers, reliability);
off-shortlist ⇒ no-evidence; Q&A never changes a vote; Quality never
aborts. (Delete-all-Part-B lives in test_two_part.py; no-hardcoded-route
in test_routing.py.)
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.runner import build_agent_prompt  # noqa: E402
from control.brain import parse_brain_order  # noqa: E402
from control.manifest import load_manifest  # noqa: E402
from control.specialists import confine_specialist  # noqa: E402
from control.schemas import VerdictEnvelope  # noqa: E402
from orchestrator import verify_receipt  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")

# 1. Chain order is always Quality → Brain-loop → Critic → Specialist →
# Referee (grep test on real call sites — anchored to assignments so
# comments/docstrings can't fake the order).
stages = ["quality = run_quality(", "order = run_brain_step(",
          "= run_critic(", "= shortlist(", "= decide(",
          "= build_report(", "append_audit({"]
positions = [orch_text.index(s) for s in stages]
check("chain-order", positions == sorted(positions), str(stages))
check("critic-once-in-chain", orch_text.count("run_critic(") == 1)

# 2. Audit carries every voice: quality, steers, per-session tools,
# critic, specialists, decision, thresholds, timings.
for key in ('"quality":', '"quality_text":', '"steers":', '"sessions":',
            '"critic":', '"specialists":', '"decision":', '"reason":',
            '"thresholds":', '"timings":'):
    check(f"audit-{key.strip(chr(34)+':')}", key in orch_text
          or key.replace('"', "'") in orch_text, key)

# 3. Per-session audit contract: steer+why, tools, re-executed numbers,
# reliability, reasoning.
for key in ("steer", "why", "suggested_subset", "tools_run", "command",
            "class", "confidence", "reliability", "reasoning"):
    check(f"session-{key}", f'"{key}"' in orch_text, key)

# 4. Autonomy: code cannot pin session tools or steers, so the same image
# MAY differ run to run. (a) verify_receipt takes no hint/subset arg —
# hints are unenforceable by construction. (b) different Brain orders
# produce different session prompts, steer text verbatim.
check("hint-unenforceable", "hint" not in inspect.signature(
    verify_receipt).parameters)
manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
from control.brain import diagnostic_pool  # noqa: E402
from control.manifest import render_subset  # noqa: E402

pool = diagnostic_pool(manifest)
diag_text = render_subset(manifest, tuple(pool))
record = {"image_path": "img.jpg", "metadata": {"age": 50}}
order_a = parse_brain_order(
    '{"steer": "Focus on the border.", "suggested_subset": '
    '["ham10000-cnn"], "stop": false, "why": "first look"}', pool)
order_b = parse_brain_order(
    '{"steer": "Verify color with fusion.", '
    '"suggested_subset": ["multimodal-fusion"], "stop": false, '
    '"why": "second angle"}', pool)
prompt_a = build_agent_prompt(record, diag_text, quality_text="Quality: pass",
                              steer=order_a["steer"],
                              hint=order_a["suggested_subset"])
prompt_b = build_agent_prompt(record, diag_text, quality_text="Quality: pass",
                              steer=order_b["steer"],
                              hint=order_b["suggested_subset"])
check("steers-flow-verbatim", "Focus on the border." in prompt_a
      and "fusion" in prompt_b)
check("same-image-may-differ", prompt_a != prompt_b)

# 5. Off-shortlist ⇒ no-evidence, on-shortlist untouched.


def env(cls, conf=0.8):
    return VerdictEnvelope(
        ran=True, command="python tools/ham10000_cnn.py --image i.jpg",
        exit_code=0, predicted_class=cls, confidence=conf,
        reasoning="r", uncertainty_flags=[])


off = confine_specialist(env("Vascular lesions"), "Melanoma",
                         ["Melanoma", "Melanocytic nevi"])
check("off-shortlist-no-evidence", off.ran is False
      and "off-shortlist" in off.reasoning, str(off))
on = confine_specialist(env("Melanoma"), "Melanoma",
                        ["Melanoma", "Melanocytic nevi"])
check("on-shortlist-passes", on.ran is True
      and on.predicted_class == "Melanoma")
dead = confine_specialist(
    VerdictEnvelope(ran=False, command="x", reasoning="r",
                    uncertainty_flags=["tool_failed"]),
    "Melanoma", ["Melanoma", "Melanocytic nevi"])
check("dead-stays-dead", dead.ran is False)

# 6. Q&A never changes a vote: ask() with a live backend leaves the
# stored decision/reason/thresholds untouched.
import skin_agent  # noqa: E402
from control.controller import CaseState, ask  # noqa: E402

_real = skin_agent.run_openclaw_cli
skin_agent.run_openclaw_cli = lambda *a, **k: {"payloads": [
    {"text": "The vote should have been benign, trust me."}]}
try:
    state = CaseState(record_index=0, report_name="r.md",
                      report_text="- Decision: **suspicious** — x",
                      receipts=[], decision="suspicious", reason="x",
                      thresholds={"borderline_threshold": 0.75})
    answer = ask(state, "Should this be benign?")
    check("qa-answers", isinstance(answer, str) and len(answer) > 0)
    check("qa-vote-stands", state.decision == "suspicious"
          and state.reason == "x"
          and state.thresholds == {"borderline_threshold": 0.75})
except Exception as exc:  # noqa: BLE001
    check("qa-answers", False, f"raised: {exc}")
    check("qa-vote-stands", False, f"raised: {exc}")
finally:
    skin_agent.run_openclaw_cli = _real

# 7. Quality never aborts: garbage backend output still yields a flagged
# verdict that flows forward (never raises, never stops the round).
from control.quality import run_quality  # noqa: E402

skin_agent.run_openclaw_cli = lambda *a, **k: {"payloads": [
    {"text": "~~ not json, just vibes ~~"}]}
try:
    q = run_quality({"image_path": "x", "metadata": {}}, "toolkit")
    check("quality-garbage-flagged", q["verdict"] == "flagged", str(q))
except Exception as exc:  # noqa: BLE001
    check("quality-garbage-flagged", False, f"raised: {exc}")
finally:
    skin_agent.run_openclaw_cli = _real

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

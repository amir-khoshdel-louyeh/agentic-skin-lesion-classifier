"""Two-part report checks (plan.tmp T8, stdlib only).
Run: .venv/bin/python tests/test_two_part.py
Exit non-zero on any failure.

Locks in §7: Part A machine/authoritative (receipts with re-executed
numbers, Quality line, referee line with rule), Part B voices with zero
vote power, delete-all-Part-B ⇒ decision unchanged, referee rule names
every path.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.controller import parse_decision  # noqa: E402
from control.decide import PART_B_HEADER, build_report, decide  # noqa: E402
from control.schemas import VerdictEnvelope  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


CANARY = "CANARY_PROSE_XYZ"


def env(cls="Melanocytic nevi", conf=0.85, tag="exact", flags=()):
    return VerdictEnvelope(
        ran=True, command="python tools/ham10000_cnn.py --image i.jpg",
        exit_code=0, predicted_class=cls, confidence=conf,
        reasoning=f"agent eloquent words {CANARY} [control: agent said "
                  f"{cls}/{conf}; reliability={tag}]",
        uncertainty_flags=list(flags))


record = {"image_path": "i.jpg", "metadata": {"age": 40}}
verdicts = [env(), env("Melanocytic nevi", 0.82, "rounded")]
decision = decide(verdicts)
quality = {"verdict": "flagged", "reasons": ["blur seen"],
           "cites": ["quality_gate: blur=low"], "cleaned_image": None}
steers = [{"session": 1, "steer": "Broad first look.",
           "suggested_subset": ["ham10000-cnn"], "stop": False,
           "why": "nothing yet"}]
sessions = [{"n": 1, "steer": "Broad first look.", "why": "nothing yet",
             "suggested_subset": ["ham10000-cnn"], "tools_run": "ham10000_cnn",
             "command": "python tools/ham10000_cnn.py --image i.jpg",
             "class": "Melanocytic nevi", "confidence": 0.85,
             "reliability": "exact",
             "reasoning": f"chose triage for speed {CANARY}"}]
critic = {"sustained": False, "reasons": []}
specialists = [{"disease": "Melanoma", "reasoning": f"advocate words {CANARY}"}]

full = build_report(record, verdicts, decision,
                    thresholds={"borderline_threshold": 0.75},
                    quality=quality, steers=steers, sessions=sessions,
                    critic=critic, specialist_texts=specialists)
part_a, part_b = full.split(PART_B_HEADER)

# 1. Part A: machine only.
check("part-a-receipt-table", "| ham10000_cnn |" in part_a
      and "0.85" in part_a and "0.82" in part_a, part_a[:400])
check("part-a-reliability", "exact" in part_a and "rounded" in part_a)
check("part-a-quality", "Quality: **flagged**" in part_a
      and "blur seen" in part_a)
check("part-a-referee", "- Decision: **benign**" in part_a
      and "rule=weighted_majority" in part_a)
check("part-a-thresholds", "borderline_threshold=0.75" in part_a)
check("part-a-no-prose", CANARY not in part_a)

# 2. Part B: voices, zero vote power.
check("part-b-note", "Part B cannot change Part A" in part_b)
check("part-b-steers", "Broad first look." in part_b
      and "nothing yet" in part_b)
check("part-b-sessions", CANARY in part_b)
check("part-b-critic", "silence" in part_b)
check("part-b-specialists", "Advocate for Melanoma" in part_b)

# 3. Delete-all-Part-B ⇒ decision unchanged.
bare = build_report(record, verdicts, decision,
                    thresholds={"borderline_threshold": 0.75},
                    quality=quality, include_part_b=False)
check("delete-part-b-identical", bare == part_a,
      "Part A differs with/without Part B")
check("decision-survives-parse",
      parse_decision(bare)[0] == parse_decision(full)[0] == "benign")

# 4. Every referee path names its rule.
check("rule-majority", decide(verdicts)["rule"] == "weighted_majority")
check("rule-malignancy", decide(
    [env("Melanoma", 0.9)])["rule"] == "malignancy_vote")
check("rule-borderline", decide(
    [env("Melanoma", 0.5)])["rule"] == "borderline_flag")
check("rule-critic", decide(
    verdicts, critic_objection=True)["rule"] == "critic_objection")
check("rule-no-evidence", decide([])["rule"] == "no_evidence")
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("rule-dilemma", '"rule": RULE_DILEMMA' in orch_text
      or "'rule': RULE_DILEMMA" in orch_text
      or "RULE_DILEMMA" in orch_text)

# 5. Audit carries per-session steer/tools/receipts/reliability.
check("audit-sessions-reliability",
      '"reliability": reliability_of' in orch_text
      or "'reliability': reliability_of" in orch_text
      or "reliability_of(verified.reasoning)" in orch_text)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

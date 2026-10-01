"""Single-writer receipt checks (plan.tmp T5, stdlib only).
Run: .venv/bin/python tests/test_single_writer.py
Exit non-zero on any failure.

Locks in: every claimed command is re-executed control-side
(verify_receipt); numbers come ONLY from re-execution; the reliability
tag is informational only, never a vote gate; advisory voices (Quality
dicts, critic text, raw tool JSON) can never become votes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.decide import decide  # noqa: E402
from control.schemas import VerdictEnvelope  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def env(cls="Melanocytic nevi", conf=0.85, tag="exact", flags=()):
    return VerdictEnvelope(
        ran=True, command="python tools/ham10000_cnn.py --image i.jpg",
        exit_code=0, predicted_class=cls, confidence=conf,
        reasoning=f"agent words [control: agent said {cls}/{conf}; "
                  f"reliability={tag}]",
        uncertainty_flags=list(flags))


# 1. Advisory output is never countable: decide() fails fast.
for bad in ({"verdict": "flagged", "reasons": ["blur"], "cites": ["c"],
             "cleaned_image": None},
            {"objection": True, "reasons": ["x"], "cites": ["1"]},
            {"status": "success", "disease_name": "Melanoma"},
            "Melanoma", None):
    try:
        decide([bad])  # type: ignore[list-item]
        check(f"rejects-{type(bad).__name__}", False, "no TypeError raised")
    except TypeError:
        check(f"rejects-{type(bad).__name__}", True)
    except Exception as exc:  # noqa: BLE001 - must be TypeError, nothing else
        check(f"rejects-{type(bad).__name__}", False, f"wrong error: {exc}")

# 2. Reliability tag never gates a vote: identical envelopes apart from
# the tag reach the identical decision.
decisions = set()
for tag in ("exact", "rounded", "fabricated", "omitted"):
    d = decide([env(tag=tag)])
    decisions.add((d["decision"], d["reason"]))
check("reliability-never-gates",
      len(decisions) == 1 and next(iter(decisions))[0] == "benign",
      str(decisions))
decide_text = (ROOT / "control" / "decide.py").read_text(encoding="utf-8")
check("decide-ignores-reliability", "reliability" not in decide_text)

# 3. Wiring: every agent claim flows through verify_receipt before it
# can vote; quality never enters the verdict list.
orch_text = (ROOT / "orchestrator.py").read_text(encoding="utf-8")
check("claims-re-executed-twice", orch_text.count(
    "verify_receipt(claimed, record, manifest)") == 2,
    "session + specialist call sites")
check("no-unverified-append",
      "verdicts.append(verified)" in orch_text
      and "verdicts.append(quality" not in orch_text
      and "verdicts.append(direct" not in orch_text
      and "verdicts.append(claimed" not in orch_text)
check("decide-gets-verdicts-only", "decide(verdicts," in orch_text)

# 4. Careful-path extra vote is single-writer by construction: built from
# the tool payload it just ran, never from agent text.
fn = orch_text.split("def run_ensemble_verdict")[1].split("\ndef ")[0]
check("ensemble-from-payload", "payload.get" in fn
      and "claimed" not in fn)

print(f"{len(failures)} failures")
raise SystemExit(1 if failures else 0)

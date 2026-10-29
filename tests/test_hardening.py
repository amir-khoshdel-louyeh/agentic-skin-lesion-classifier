"""Hardening checks for the E2E defects found on 2026-10-20 (stdlib only).

Run: .venv/bin/python -m pytest tests/test_hardening.py

Locks in five fixes that the round_1_20261020 run exposed:

1. Receipts: a session's 1..n tool commands are ALL receipted, one row per
   distinct tool. Running the same model twice is one vote, not two, and a
   second model's disagreement cannot disappear behind the first.
2. Flags: `uncertainty_flags` come from the re-executed tool JSON. An
   agent-declared "borderline" lives in `agent_flags` and never votes.
3. Quality: the control gate is rendered in Part A, an agent `pass` on a
   failed gate is overridden, and the report footer never lands in a
   verdict field.
4. Brain: numeric scores are stripped from the steer a session sees, and
   pre-rendered tool lines use absolute paths.
5. Part A carries provenance (image sha256 + backend).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.brain import (  # noqa: E402
    build_brain_prompt,
    diagnostic_pool,
    parse_brain_order,
    sanitize_steer,
)
from control.critic import build_critic_prompt  # noqa: E402
from control.decide import build_report, decide, PART_B_HEADER  # noqa: E402
from control.manifest import load_manifest, render_subset_for_case  # noqa: E402
from control.jsontext import first_json_object  # noqa: E402
from control.quality import (  # noqa: E402
    gate_cite,
    parse_quality,
    quality_binds,
    resolve_cleaned_image,
)
from control.schemas import VerdictEnvelope  # noqa: E402
from control.specialists import shortlist  # noqa: E402
from orchestrator import verify_receipt, verify_receipts  # noqa: E402

def test_hardening_checks():
    failures = []


    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name
              + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)


    def env(tool="ham10000_cnn", cls="Melanoma", conf=0.9, flags=(),
            agent_flags=(), image="i.jpg", extra_commands=None):
        return VerdictEnvelope(
            ran=True,
            command=f"python tools/{tool}.py --image {image}",
            commands=list(extra_commands or []),
            exit_code=0, predicted_class=cls, confidence=conf,
            reasoning="t [control: agent said None/None; reliability=omitted]",
            uncertainty_flags=list(flags), agent_flags=list(agent_flags))


    # 1. Multi-command receipts / one vote per tool.
    single = env(extra_commands=[])
    check("command-list-single", single.command_list() == [single.command],
          str(single.command_list()))
    multi = env(extra_commands=["python tools/isic_vit.py --image i.jpg",
                                "python tools/isic_vit.py --image i.jpg"])
    check("command-list-dedupes",
          multi.command_list() == [multi.command, 
                                   "python tools/isic_vit.py --image i.jpg"],
          str(multi.command_list()))

    same_tool_twice = [env("isic_vit", "Melanoma", 0.87),
                       env("isic_vit", "Melanoma", 0.87)]
    deduped = decide(same_tool_twice)
    check("one-tool-votes-once",
          deduped["verdicts_counted"] == 2 and deduped["distinct_tools"] == 1,
          str(deduped))

    disagreeing = [env("isic_vit", "Melanoma", 0.90),
                   env("ham10000_cnn", "Benign keratosis", 0.90)]
    check("two-tools-two-votes", decide(disagreeing)["distinct_tools"] == 2)
    check("hidden-disagreement-would-shortlist",
          shortlist(disagreeing) == ["Melanoma", "Benign keratosis"],
          str(shortlist(disagreeing)))

    # 2. Flags are control-side; agent flags never vote.
    agent_only = [env("ham10000_cnn", "Melanocytic nevi", 0.95,
                      agent_flags=("borderline",))]
    check("agent-flag-is-advisory",
          decide(agent_only)["rule"] == "weighted_majority", str(agent_only[0]))
    tool_flag = [env("isic_vit", "Melanocytic nevi", 0.95,
                     flags=("borderline",))]
    check("tool-flag-counts",
          decide(tool_flag)["rule"] == "borderline_flag", str(decide(tool_flag)))
    # A flagged image can never come out "benign", no matter how confident the
    # models are: the deterministic quality verdict binds.
    confident_benign = [env("isic_vit", "Melanocytic nevi", 0.97)]
    check("quality-flag-never-benign",
          decide(confident_benign, quality_flagged=True)["rule"]
          == "quality_flagged", str(decide(confident_benign,
                                           quality_flagged=True)))
    check("quality-pass-allows-benign",
          decide(confident_benign, quality_flagged=False)["rule"]
          == "weighted_majority")
    # Machine rules outrank the LLM critic, so Part A always prints a
    # reproducible reason when one exists.
    check("machine-rule-precedes-critic",
          decide([env("isic_vit", "Melanoma", 0.9)], critic_objection=True,
                 quality_flagged=True)["rule"] == "malignancy_vote")
    check("quality-precedes-critic",
          decide(confident_benign, critic_objection=True,
                 quality_flagged=True)["rule"] == "quality_flagged")
    check("critic-still-can-flag",
          decide(confident_benign, critic_objection=True)["rule"]
          == "critic_objection")
    # Doctor thresholds actually bind (they used to be print-only).
    check("confidence-floor-binds",
          decide([env("isic_vit", "Melanocytic nevi", 0.80)],
                 confidence_threshold=0.85)["rule"] == "confidence_floor")
    check("confidence-floor-default-noop",
          decide([env("isic_vit", "Melanocytic nevi", 0.80)])["rule"]
          == "weighted_majority")

    import os  # noqa: E402
    from control.confidence import ENTROPY_ENV, env_entropy_threshold  # noqa: E402

    _prev = os.environ.get(ENTROPY_ENV)
    try:
        os.environ[ENTROPY_ENV] = "0.4"
        check("entropy-threshold-from-env", env_entropy_threshold() == 0.4)
        os.environ[ENTROPY_ENV] = "nonsense"
        check("entropy-threshold-falls-back",
              env_entropy_threshold() == 0.9)
    finally:
        if _prev is None:
            os.environ.pop(ENTROPY_ENV, None)
        else:
            os.environ[ENTROPY_ENV] = _prev

    # 3. Quality: footer stripped, control gate wins a contradictory `pass`.
    footered = json.dumps({
        "verdict": "pass",
        "reasons": ["Blur looks acceptable. Research use only — not a medical "
                    "diagnosis."],
        "cites": ["quality_gate: looks fine. Research use only — not a medical "
                  "diagnosis."],
        "cleaned_image": None,
    })
    gate = {"passed": False, "flags": ["blurry"],
            "metrics": {"blur_variance": 24.36}, "error": ""}
    parsed = parse_quality(footered, gate_evidence=gate)
    check("quality-footer-stripped",
          all("Research use only" not in text
              for text in parsed["reasons"] + parsed["cites"]), str(parsed))
    check("quality-pass-overridden", parsed["verdict"] == "flagged",
          str(parsed))
    check("quality-control-cite",
          bool(parsed.get("control_cites"))
          and "24.36" in parsed["control_cites"][0], str(parsed.get(
              "control_cites")))
    legit = parse_quality(json.dumps({
        "verdict": "flagged", "reasons": ["blur_variance 24.36 < 60"],
        "cites": ["quality_gate: blur_variance=24.36"], "cleaned_image": None,
    }), gate_evidence=gate)
    check("quality-legit-verdict-kept", legit["verdict"] == "flagged")

    # 4. Brain: no numeric scores reach a session; tool lines are absolute.
    check("steer-strips-scores",
          "0.8715" not in sanitize_steer("Session 1 yielded Melanoma at 0.8715 "
                                         "with 87% confidence"),
          sanitize_steer("Session 1 yielded Melanoma at 0.8715"))
    ordered = parse_brain_order(json.dumps({
        "steer": "Session 1 yielded Melanoma with 0.8715 confidence; verify.",
        "suggested_subset": ["ham10000-cnn"], "stop": False, "why": "verify",
    }), ["ham10000-cnn"])
    check("parsed-steer-sanitized", "0.8715" not in ordered["steer"],
          ordered["steer"])
    critic_prompt = build_critic_prompt(
        [env("isic_vit", "Melanoma", 0.87)], ["blurry"], "Quality: flagged")
    check("critic-quality-is-handled-elsewhere",
          "ALREADY handled deterministically" in critic_prompt
          and "empty" in critic_prompt, critic_prompt[:300])
    check("steer-drops-report-footer",
          "Research use only" not in sanitize_steer(
              "Verify the border.\nResearch use only — not a medical "
              "diagnosis."), sanitize_steer("Verify. Research use only — "
                                           "not a medical diagnosis."))

    manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
    pool = diagnostic_pool(manifest)
    record = {"image_path": "dataset/ISIC2019_test/ISIC_0000002.jpg",
              "metadata": {"patient_id": "A001"}}
    rendered = render_subset_for_case(manifest, tuple(pool), record)
    command_lines = [line.split("`")[1] for line in rendered.splitlines()
                     if line.startswith("- ")]
    check("tool-lines-absolute",
          bool(command_lines) and all(str(ROOT) in cmd for cmd in command_lines),
          str(command_lines[:2]))

    exhausted = build_brain_prompt(
        record, "Quality: flagged",
        [{"n": 1, "class": "Melanoma", "confidence": 0.9, "flags": "",
          "tools": "+".join(name.replace("-", "_") for name in pool),
          "results": "", "reasoning": ""}],
        "role text", pool)
    check("brain-knows-pool-exhausted",
          "adds no new evidence" in exhausted, exhausted[-300:])

    # 5. Part A: agent-flag column, control gate line, provenance.
    report = build_report(
        {**record, "image_path": "dataset/ISIC2019_test/ISIC_0000002.jpg"},
        [env("isic_vit", "Melanoma", 0.87, agent_flags=("borderline",))],
        decide([env("isic_vit", "Melanoma", 0.87)]),
        thresholds={"borderline_threshold": 0.75},
        quality={"verdict": "flagged", "reasons": ["blur"],
                 "cites": ["quality_gate: blur_variance=24.36"],
                 "control_cites": ['quality_gate: {"quality_pass": false}'],
                 "cleaned_image": None},
        provenance={"image_sha256": "abc123", "backend": "openclaw agent=main",
                    "started_at": "2026-10-20T00:00:00+00:00"},
    )
    part_a = report.split(PART_B_HEADER)[0]
    check("part-a-agent-flag-column",
          "flags (agent)" in part_a and "borderline" in part_a, part_a[:600])
    check("part-a-control-gate",
          "Gate (control)" in part_a and "quality_pass" in part_a, part_a[:600])
    check("part-a-provenance",
          "abc123" in part_a and "openclaw agent=main" in part_a, part_a[:400])
    check("part-a-distinct-tools", "distinct tools" in part_a, part_a[:600])

    # 6. Integration: the tool's own flags (not the agent's) become the receipt.
    record_real = {"image_path": "dataset/ISIC2019_test/ISIC_0000000.jpg",
                   "metadata": {}}
    claimed = VerdictEnvelope(
        ran=True,
        command=("python tools/ham10000_cnn.py --image "
                 "dataset/ISIC2019_test/ISIC_0000000.jpg"),
        exit_code=0, reasoning="t", uncertainty_flags=["borderline"])
    verified = verify_receipt(claimed, record_real, manifest)
    check("receipt-flags-from-tool",
          verified.ran and verified.uncertainty_flags == []
          and verified.agent_flags == ["borderline"], str(verified))

    # An agent naming a class the tool did not produce is recorded as a
    # class_mismatch (display only — the tool's class still wins).
    wrong_claim = VerdictEnvelope(
        ran=True,
        command=("python tools/ham10000_cnn.py --image "
                 "dataset/ISIC2019_test/ISIC_0000000.jpg"),
        exit_code=0, predicted_class="Benign keratosis",
        reasoning="t", uncertainty_flags=[])
    mismatched = verify_receipt(wrong_claim, record_real, manifest)
    check("class-mismatch-recorded",
          mismatched.predicted_class == verified.predicted_class
          and "reliability=class_mismatch" in mismatched.reasoning,
          str(mismatched))

    claimed_two = VerdictEnvelope(
        ran=True,
        command=("python tools/ham10000_cnn.py --image "
                 "dataset/ISIC2019_test/ISIC_0000000.jpg"),
        commands=[
            "python tools/ham10000_cnn.py --image "
            "dataset/ISIC2019_test/ISIC_0000000.jpg",
            "python tools/isic_vit.py --image "
            "dataset/ISIC2019_test/ISIC_0000000.jpg",
        ],
        exit_code=0, reasoning="t", uncertainty_flags=[])
    rows = verify_receipts(claimed_two, record_real, manifest)
    check("receipts-one-row-per-tool",
          len(rows) == 2 and {row.command.split()[1] for row in rows} == {
              "tools/ham10000_cnn.py", "tools/isic_vit.py"}, str(rows))

    # 7. JSON extraction stops at the first complete object (a reply with a
    # second object or trailing prose used to lose the whole verdict).
    two_objects = ('{"a": 1}\n{"b": 2}')
    check("json-first-object", first_json_object(two_objects) == {"a": 1},
          str(first_json_object(two_objects)))
    braced_prose = 'note {not json here} then {"verdict": "pass"}'
    check("json-skips-invalid-braces",
          first_json_object(braced_prose) == {"verdict": "pass"},
          str(first_json_object(braced_prose)))
    check("json-none-when-absent", first_json_object("no braces at all") is None)
    check("json-none-when-truncated", first_json_object('{"ran": true') is None)

    envelope_twice = (
        '{"ran": true, "command": "python tools/ham10000_cnn.py --image i.jpg",'
        ' "exit_code": 0, "predicted_class": null, "confidence": null,'
        ' "reasoning": "r", "uncertainty_flags": []}\n{"ran": false}')
    from control.enforcement import parse_envelope  # noqa: E402

    check("envelope-survives-second-object",
          parse_envelope(envelope_twice).ran is True)
    quality_twice = json.dumps({
        "verdict": "flagged", "reasons": ["blur"],
        "cites": ["quality_gate: x"], "cleaned_image": None,
    }) + '\n{"unexpected": true}'
    check("quality-survives-second-object",
          parse_quality(quality_twice)["verdict"] == "flagged")
    brain_twice = json.dumps({
        "steer": "verify", "suggested_subset": [], "stop": False,
        "why": "x",
    }) + '\n{"unexpected": true}'
    check("brain-survives-second-object",
          parse_brain_order(brain_twice, [])["stop"] is False)

    # 8. The round binds on the gate of the ACTIVE image, not on the label:
    # a `cleaned` verdict over a still-failing gate must still bind.
    check("quality-binds-cleaned-gate-failed",
          quality_binds({"verdict": "cleaned"}, {"passed": False}) is True)
    check("quality-binds-cleaned-gate-ok",
          quality_binds({"verdict": "cleaned"}, {"passed": True}) is False)
    check("quality-binds-agent-flag",
          quality_binds({"verdict": "flagged"}, {"passed": True}) is True)
    check("quality-pass-gate-ok-is-free",
          quality_binds({"verdict": "pass"}, {"passed": True}) is False)

    # 9. A model-authored cleaned path is not evidence: only the offered
    # output path may become the active image.
    import tempfile  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        offered = Path(tmp) / "offered_cleaned.jpg"
        other = Path(tmp) / "other_case.jpg"
        offered.write_bytes(b"offered")
        other.write_bytes(b"other")
        cleaned = {"verdict": "cleaned", "cleaned_image": str(other)}
        check("cleaned-path-must-match-offered",
              resolve_cleaned_image({}, cleaned, expected=str(offered)) is None)
        cleaned_ok = {"verdict": "cleaned", "cleaned_image": str(offered)}
        check("cleaned-path-accepted",
              resolve_cleaned_image({}, cleaned_ok,
                                    expected=str(offered)) == offered)
        check("cleaned-missing-file-rejected",
              resolve_cleaned_image(
                  {}, {"verdict": "cleaned",
                       "cleaned_image": str(Path(tmp) / "nope.jpg")},
                  expected=str(Path(tmp) / "nope.jpg")) is None)
    check("gate-cite-labels-active",
          gate_cite({"passed": False}, label="active")
          .startswith("quality_gate@active:"))

    # 10. OpenClaw retries: narrow transient matching + a fresh session id
    # per attempt (the retry targets session-state errors).
    import skin_agent  # noqa: E402
    from skin_agent import is_transient_openclaw_error  # noqa: E402

    check("retry-not-on-generic-threshold",
          is_transient_openclaw_error("confidence threshold not met") is False)
    check("retry-not-on-bare-429",
          is_transient_openclaw_error("counter mismatch 1429") is False)
    check("retry-on-http-429",
          is_transient_openclaw_error("HTTP 429 Too Many Requests") is True)

    calls = []


    class _Fail:
        returncode = 1
        stdout = ""
        stderr = "HTTP 429 Too Many Requests"


    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Fail()


    _real_run = skin_agent.subprocess.run
    _real_sleep = skin_agent.time.sleep
    _real_which = skin_agent.find_openclaw_executable
    skin_agent.subprocess.run = _fake_run
    skin_agent.time.sleep = lambda _s: None
    skin_agent.find_openclaw_executable = lambda: "openclaw"
    try:
        try:
            skin_agent.run_openclaw_cli("hi", show_command=False)
        except RuntimeError:
            pass
    finally:
        skin_agent.subprocess.run = _real_run
        skin_agent.time.sleep = _real_sleep
        skin_agent.find_openclaw_executable = _real_which
    session_ids = [cmd[cmd.index("--session-id") + 1] for cmd in calls]
    check("retry-fresh-session-each-attempt",
          len(session_ids) == 6 and len(set(session_ids)) == 6,
          str(session_ids))

    assert not failures, f"{len(failures)} failures: {failures}"

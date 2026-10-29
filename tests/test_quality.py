"""Quality Agent checks (plan.tmp T2, stdlib only).
Run: .venv/bin/python -m pytest tests/test_quality.py

Locks in §2.1: pass|cleaned|flagged schema with cites, never-diagnose /
never-stop rules, fail-safe to flagged (never an abort), quality skill
toolkit, and chain-step-1 wiring in the orchestrator.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.manifest import (  # noqa: E402
    load_manifest,
    render_subset,
    render_subset_for_case,
)
from control.prompts import load_role  # noqa: E402
from control.quality import (  # noqa: E402
    FAILSAFE_CITE,
    MAX_REPAIR_RETRIES,
    QUALITY_TOOLKIT,
    build_quality_prompt,
    build_quality_repair_prompt,
    parse_quality,
    quality_line,
    resolve_cleaned_image,
    run_quality,
)

def test_quality_checks():
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

    # 3b. Exactly one repair retry on parse failure: garbage then valid
    # recovers; garbage twice stays fail-safe flagged. Genuine flagged
    # verdicts (real cites) never trigger a retry.
    calls = []


    def _garbage_then_valid(*args, **kwargs):
        calls.append(args[0][:40])
        if len(calls) == 1:
            return {"payloads": [{"text": "~~ not json ~~"}]}
        return {"payloads": [{"text": json.dumps(
            {"verdict": "pass", "reasons": ["sharp"],
             "cites": ["quality_gate: pass"], "cleaned_image": None})}]}


    skin_agent.run_openclaw_cli = _garbage_then_valid
    try:
        out = run_quality({"image_path": "x.jpg", "metadata": {}},
                          "toolkit", agent_id="main")
        check("repair-retry-recovers", out["verdict"] == "pass"
              and len(calls) == 2, f"{out} calls={len(calls)}")
    except Exception as exc:  # noqa: BLE001
        check("repair-retry-recovers", False, f"raised: {exc}")
    finally:
        skin_agent.run_openclaw_cli = _real


    garbage_calls = []


    def _always_garbage(*args, **kwargs):
        garbage_calls.append(args[0][:40])
        return {"payloads": [{"text": "~~ not json ~~"}]}


    skin_agent.run_openclaw_cli = _always_garbage
    try:
        out = run_quality({"image_path": "x.jpg", "metadata": {}},
                          "toolkit", agent_id="main")
        check("double-garbage-flagged", out["verdict"] == "flagged"
              and out["cites"] == [FAILSAFE_CITE], str(out))
        # Retries stay bounded: one initial session + MAX_REPAIR_RETRIES.
        check("repair-retries-bounded", MAX_REPAIR_RETRIES == 2,
              str(MAX_REPAIR_RETRIES))
        check("repair-retries-used",
              len(garbage_calls) == MAX_REPAIR_RETRIES + 1,
              f"sessions={len(garbage_calls)}")
    except Exception as exc:  # noqa: BLE001
        check("double-garbage-flagged", False, f"raised: {exc}")
    finally:
        skin_agent.run_openclaw_cli = _real

    check("repair-prompt-schema",
          "pass|cleaned|flagged" in build_quality_repair_prompt("garbage"))
    check("repair-prompt-names-defect",
          "claimed cleaned without a cleaned_image path" in
          build_quality_repair_prompt(
              "garbage", ["claimed cleaned without a cleaned_image path"]))
    check("quality-no-fake-cleaning",
          "did not perform" in load_role("quality")
          or "did not clean" in load_role("quality"))

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
    check("prompt-one-json-object", "exactly one JSON object" in prompt)
    check("prompt-never-looks-at-image", "never look at the image" in prompt)
    check("prompt-no-evidence-by-default", "quality_pass" not in prompt)

    # 5b. Control-side gate numbers are handed over as EVIDENCE (never a
    # decision): the agent still chooses the verdict, and a small local model
    # no longer has to execute the command to cite a real number.
    gate = {"passed": False, "metrics": {"blur_variance": 12.5},
            "flags": ["blurry"], "error": ""}
    evidence_prompt = build_quality_prompt(
        {"image_path": "img.jpg", "metadata": {}}, role_text, toolkit_text,
        gate_evidence=gate)
    check("prompt-carries-gate-evidence",
          "blur_variance" in evidence_prompt and "blurry" in evidence_prompt
          and "read-only evidence" in evidence_prompt)
    check("prompt-evidence-keeps-decision-with-agent",
          "quality_pass" in evidence_prompt
          and "exactly one JSON object" in evidence_prompt
          and "pass|cleaned|flagged" in evidence_prompt)
    check("prompt-evidence-reports-tool-error",
          "tool_error" in build_quality_prompt(
              {"image_path": "img.jpg", "metadata": {}}, role_text,
              toolkit_text,
              gate_evidence={"passed": False, "metrics": {}, "flags": [],
                             "error": "tool_failed"}))

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
    # Chain step 1 must get the same pre-filled EXACT lines the diagnostic
    # sessions get: the Quality Agent's hand-built paths (placeholder
    # <path>, then a workspace-absolute path) were its observed failure mode.
    check("orchestrator-quality-exact-lines",
          "render_subset_for_case(manifest, QUALITY_TOOLKIT" in orch_text,
          "Quality Agent still receives placeholder <path> lines")
    quality_case_text = render_subset_for_case(
        manifest, QUALITY_TOOLKIT,
        {"image_path": "dataset/ISIC2019_test/ISIC_0000002.jpg",
         "metadata": {"age": 60}})
    quality_lines = [line for line in quality_case_text.splitlines()
                     if line.startswith("- ")]
    check("quality-lines-no-placeholders",
          quality_lines and all("<path" not in line for line in quality_lines),
          quality_case_text)
    check("quality-lines-real-image",
          "dataset/ISIC2019_test/ISIC_0000002.jpg" in quality_case_text)
    check("quality-preprocess-has-output", "--output" in quality_case_text,
          quality_case_text)
    check("orchestrator-quality-audit", '"quality": quality' in orch_text
          or "'quality': quality" in orch_text or '"quality":' in orch_text)
    check("orchestrator-quality-timing", '"quality_s"' in orch_text
          or "'quality_s'" in orch_text)
    # The deterministic gate must run BEFORE the Quality Agent so its numbers
    # can be handed over as evidence.
    check("orchestrator-gate-before-quality",
          0 <= orch_text.index("run_guard(") < orch_text.index("run_quality("),
          "gate must be collected before the Quality Agent runs")
    check("orchestrator-quality-gets-evidence",
          "gate_evidence=guard" in orch_text)
    check("quality-line-renders", quality_line(
        {"verdict": "flagged", "reasons": ["blur"],
         "cites": ["c"], "cleaned_image": None}).startswith("Quality: flagged"))

    # 8. Cleaned-image hand-off (§2.1): usable only when the file exists.
    import tempfile  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="qclean_") as tmp:
        root = Path(tmp)
        real = root / "cleaned.jpg"
        real.write_bytes(b"j")
        record = {"image_path": "x.jpg", "metadata": {}}
        check("cleaned-existing-resolves",
              resolve_cleaned_image(record,
                                    {"verdict": "cleaned",
                                     "cleaned_image": str(real)},
                                    root=root) == real)
        check("cleaned-relative-resolves",
              resolve_cleaned_image(record,
                                    {"verdict": "cleaned",
                                     "cleaned_image": "cleaned.jpg"},
                                    root=root) == real)
        check("cleaned-missing-is-none",
              resolve_cleaned_image(record,
                                    {"verdict": "cleaned",
                                     "cleaned_image": str(root / "nope.jpg")},
                                    root=root) is None)
        for verdict in ("pass", "flagged"):
            check(f"{verdict}-is-none",
                  resolve_cleaned_image(record,
                                        {"verdict": verdict,
                                         "cleaned_image": str(real)},
                                        root=root) is None)
    check("cleaned-empty-is-none",
          resolve_cleaned_image({"image_path": "x.jpg"},
                                {"verdict": "cleaned",
                                 "cleaned_image": "  "}) is None)
    check("orchestrator-uses-cleaned", "resolve_cleaned_image(" in orch_text
          and '"cleaned_image"' in orch_text)
    check("orchestrator-active-record", "active_record" in orch_text
          and orch_text.count("verify_receipts(c, active_record, manifest)") == 2)

    assert not failures, f"{len(failures)} failures: {failures}"

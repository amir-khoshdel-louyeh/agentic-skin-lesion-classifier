"""Hermetic end-to-end orchestration test.

The real run_round loop, runner/session repair path, receipt verifier,
referee and report writer are exercised. Only the external model/tool
boundaries are replaced: OpenClaw becomes a scripted CLI, the quality and
brain agents are fixed responses, and receipt execution returns known JSON.
No Ollama, OpenClaw, GPU or model checkpoint is used.
"""

import hashlib
import json
from pathlib import Path

import pytest

import orchestrator
from control.manifest import load_manifest
from control.schemas import VerdictEnvelope


class Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _envelope(command, *, ran=True, cls=None, confidence=None):
    return {
        "ran": ran,
        "command": command,
        "exit_code": 0 if ran else 1,
        "predicted_class": cls,
        "confidence": confidence,
        "reasoning": "scripted session",
        "uncertainty_flags": [],
    }


def test_run_round_is_hermetic_and_rejects_unreceipted_claims(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parent.parent
    image = tmp_path / "case.jpg"
    image.write_bytes(b"temporary image")
    image_path = str(image.resolve())
    report_dir = tmp_path / "report"

    manifest = load_manifest(root / "tools" / "manifest.yaml")
    tool = manifest.by_name("ham10000-cnn")
    good_command = f"python {tool.command} --image {image_path}"
    bad_command = f"python tools/quality_gate.py --image {image_path}"

    monkeypatch.setattr(orchestrator, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "startup_path_check", lambda: None)
    monkeypatch.setattr(
        orchestrator, "effective_thresholds",
        lambda _role: {
            "entropy_threshold": 0.9,
            "borderline_threshold": 0.75,
            "confidence_threshold": 0.75,
        },
    )
    monkeypatch.setattr(
        orchestrator, "load_record",
        lambda _path, _index: {
            "image_path": image_path,
            "metadata": {"age": 50},
        },
    )
    monkeypatch.setattr(
        orchestrator, "diagnostic_pool", lambda _manifest: [tool.name]
    )
    monkeypatch.setattr("control.vram.unload_ollama", lambda: False)
    monkeypatch.setattr(
        orchestrator, "run_guard",
        lambda _image: {"passed": True, "metrics": {}, "flags": [], "error": ""},
    )
    monkeypatch.setattr(
        orchestrator, "run_quality",
        lambda *_args, **_kwargs: {
            "verdict": "pass",
            "reasons": ["sharp"],
            "cites": ["quality_gate: pass"],
            "cleaned_image": None,
        },
    )
    monkeypatch.setattr(
        orchestrator, "run_abcde",
        lambda _image: {"risk_band": "low", "score": 0.0},
    )
    monkeypatch.setattr(
        orchestrator, "run_critic", lambda *_args, **_kwargs: (False, [])
    )
    monkeypatch.setattr(orchestrator, "ping", lambda *_args, **_kwargs: None)

    orders = [
        {"steer": "first look", "suggested_subset": [tool.name],
         "stop": False, "why": "first"},
        {"steer": "second look", "suggested_subset": [tool.name],
         "stop": False, "why": "second"},
        {"steer": "stop", "suggested_subset": [],
         "stop": True, "why": "enough"},
    ]
    calls = {"brain": 0}

    def brain(*_args, **_kwargs):
        order = orders[min(calls["brain"], len(orders) - 1)]
        calls["brain"] += 1
        return order

    monkeypatch.setattr(orchestrator, "run_brain_step", brain)

    # First session: a rejected claim, then its rejected repair. Second
    # session: a valid claim. The real run_session repair path is used.
    cli_calls = []

    def fake_cli(prompt, **_kwargs):
        cli_calls.append(prompt)
        if "REJECTED" in prompt or len(cli_calls) == 1:
            claim = _envelope(bad_command, cls="Melanoma", confidence=0.99)
        else:
            claim = _envelope(good_command, cls="Melanoma", confidence=0.99)
        return {"payloads": [{"text": json.dumps(claim)}]}

    import agents.runner as runner
    monkeypatch.setattr(runner, "run_openclaw_cli", fake_cli)

    tool_calls = []

    def fake_tool(cmd, **_kwargs):
        tool_calls.append(cmd)
        return Proc(stdout=json.dumps({
            "status": "success",
            "disease_name": "Melanocytic nevi",
            "confidence_score": 0.88,
            "uncertainty_flags": [],
        }))

    audits = []
    monkeypatch.setattr(
        orchestrator, "append_audit",
        lambda entry: audits.append(entry),
    )
    stages = []

    report_path = orchestrator.run_round(
        0,
        prompt_file="prompt.yaml",
        receipt_runner=fake_tool,
        on_stage=lambda stage, payload: stages.append((stage, payload)),
    )

    assert report_path == report_dir / report_path.name
    report = report_path.read_text(encoding="utf-8")
    assert "- Decision: **benign**" in report
    assert "Melanocytic nevi" in report
    assert "quality_gate.py" in report
    assert report_path.parent == report_dir

    # The unexecuted excluded tool never reached the executor; the valid
    # tool was executed exactly once and supplied the report's numbers.
    assert len(tool_calls) == 1
    assert "ham10000_cnn.py" in " ".join(tool_calls[0])

    assert len(audits) == 1
    audit = audits[0]
    assert audit["decision"] == "benign"
    assert len(audit["sessions"]) == 2
    assert audit["sessions"][0]["class"] is None
    assert audit["sessions"][1]["class"] == "Melanocytic nevi"
    assert audit["sessions"][1]["confidence"] == 0.88
    assert audit["sessions"][0]["per_tool"][0]["class"] is None
    assert audit["sessions"][1]["per_tool"][0]["class"] == "Melanocytic nevi"

    ledger = audit["receipts"]
    assert len(ledger) == 1
    receipt = ledger[0]
    assert receipt["tool"] == "ham10000-cnn"
    assert receipt["exit_code"] == 0
    assert receipt["duration_ms"] is not None and receipt["duration_ms"] >= 0
    assert receipt["image_sha256"] == hashlib.sha256(b"temporary image").hexdigest()
    assert len(receipt["receipt_id"]) == 64
    assert receipt["receipt_id"] == audit["sessions"][1]["per_tool"][0]["receipt_id"]
    assert "Receipt ledger (content hashes)" in report
    assert receipt["receipt_id"] in report
    assert receipt["stdout_sha256"] in report

    emitted = [stage for stage, _payload in stages]
    assert {"quality", "brain", "session", "critic", "referee"} <= set(emitted)

    event_path = tmp_path / "report" / "events.jsonl"
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events
    assert {event["run_id"] for event in events} == {audit["run_id"]}
    assert {"guard", "quality", "brain", "session", "critic",
            "referee", "round_end"} <= {event["stage"] for event in events}
    assert any(event["stage"] == "referee"
               and event["decision"] == "benign" for event in events)
    assert all({"event_id", "run_id", "ts", "stage", "status"}
               <= set(event) for event in events)
    assert not any("steer" in event for event in events)
    assert f"- Run id: {audit['run_id']}" in report

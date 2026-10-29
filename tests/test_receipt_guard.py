"""Deterministic receipt verification with an injected tool executor."""

import hashlib
import json
import shlex
import sys
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


@pytest.fixture
def harness(tmp_path, monkeypatch):
    image = tmp_path / "case.jpg"
    image.write_bytes(b"image")
    manifest = load_manifest(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "tools" / "manifest.yaml"
    )
    tool = manifest.by_name("ham10000-cnn")
    monkeypatch.setattr(
        orchestrator, "diagnostic_pool", lambda _manifest: [tool.name]
    )
    monkeypatch.setattr("control.vram.unload_ollama", lambda: False)
    record = {"image_path": str(image), "metadata": {}}
    command = f"python {tool.command} --image {image}"
    return record, manifest, command


def claim(command, predicted_class=None, confidence=None):
    return VerdictEnvelope(
        ran=True,
        command=command,
        exit_code=0,
        predicted_class=predicted_class,
        confidence=confidence,
        reasoning="agent claim",
        uncertainty_flags=["borderline"],
    )


def test_receipt_values_come_from_execution_not_agent_prose(harness):
    record, manifest, command = harness
    calls = []

    def runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Proc(
            stdout=json.dumps({
                "status": "success",
                "disease_name": "Melanoma",
                "confidence_score": 0.91,
                "uncertainty_flags": ["borderline"],
            })
        )

    result = orchestrator.verify_receipt(
        claim(command, "Benign keratosis", 0.4),
        record, manifest, runner=runner,
    )

    assert len(calls) == 1
    assert result.ran is True
    assert result.predicted_class == "Melanoma"
    assert result.confidence == 0.91
    assert "reliability=class_mismatch" in result.reasoning
    assert set(result.uncertainty_flags) == {"borderline", "disagreement"}
    assert result.agent_flags == ["borderline"]


def test_unavailable_tool_is_rejected_without_execution(harness):
    record, manifest, _command = harness
    image = record["image_path"]
    calls = []

    result = orchestrator.verify_receipt(
        claim(f"python tools/quality_gate.py --image {image}"),
        record, manifest,
        runner=lambda *args, **kwargs: calls.append(args) or Proc(),
    )

    assert calls == []
    assert result.ran is False
    assert result.predicted_class is None
    assert result.confidence is None
    assert "diagnostic pool" in result.reasoning


def test_failed_or_invalid_execution_never_becomes_evidence(harness):
    record, manifest, command = harness

    failed = orchestrator.verify_receipt(
        claim(command, "Melanoma", 0.9),
        record, manifest,
        runner=lambda *args, **kwargs: Proc(returncode=2, stderr="crashed"),
    )
    assert failed.ran is False
    assert failed.predicted_class is None
    assert "re-execution exit 2" in failed.reasoning

    invalid = orchestrator.verify_receipt(
        claim(command, "Melanoma", 0.9),
        record, manifest,
        runner=lambda *args, **kwargs: Proc(stdout="not json"),
    )
    assert invalid.ran is False
    assert invalid.confidence is None
    assert "no JSON" in invalid.reasoning


def test_image_mismatch_is_rejected_before_execution(harness):
    record, manifest, _command = harness
    calls = []

    result = orchestrator.verify_receipt(
        claim("python tools/ham10000_cnn.py --image /tmp/other.jpg"),
        record, manifest,
        runner=lambda *args, **kwargs: calls.append(args) or Proc(),
    )

    assert calls == []
    assert result.ran is False
    assert "image path does not match" in result.reasoning


def test_every_claimed_tool_gets_its_own_receipt(harness):
    record, manifest, command = harness
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return Proc(
            stdout=json.dumps({
                "status": "success",
                "disease_name": "Melanoma",
                "confidence_score": 0.8,
                "uncertainty_flags": [],
            })
        )

    claimed = claim(command)
    claimed = claimed.model_copy(update={
        "commands": [
            command,
            f"python tools/quality_gate.py --image {record['image_path']}",
        ]
    })
    rows = orchestrator.verify_receipts(
        claimed, record, manifest, runner=runner
    )

    assert len(rows) == 2
    assert sum(row.ran for row in rows) == 1
    assert len(calls) == 1
    assert {row.ran for row in rows} == {True, False}


@pytest.mark.parametrize("payload", [
    {"status": "success", "disease_name": "Melanoma",
     "confidence_score": "0.9"},
    {"status": "success", "disease_name": "Melanoma",
     "confidence_score": 1.2},
    {"status": "success", "disease_name": "Not a class",
     "confidence_score": 0.9},
    {"status": "success", "disease_name": "Melanoma"},
    {"status": "success", "confidence_score": 0.9},
    {"status": "success", "disease_name": "Melanoma",
     "confidence_score": 0.9, "uncertainty_flags": "borderline"},
    {"status": "success", "disease_name": "Melanoma",
     "confidence_score": 0.9, "predicted_class_index": "1"},
])
def test_malformed_tool_json_never_becomes_evidence(harness, payload):
    record, manifest, command = harness
    result = orchestrator.verify_receipt(
        claim(command, "Melanoma", 0.9),
        record, manifest,
        runner=lambda *args, **kwargs: Proc(stdout=json.dumps(payload)),
    )

    assert result.ran is False
    assert result.predicted_class is None
    assert result.confidence is None
    assert "invalid tool JSON" in result.reasoning


def test_receipt_ledger_hashes_the_exact_execution(harness):
    record, manifest, command = harness
    stdout = json.dumps({
        "status": "success",
        "disease_name": "Melanoma",
        "confidence_score": 0.91,
        "uncertainty_flags": [],
    })
    result = orchestrator.verify_receipt(
        claim(command, "Melanoma", 0.91),
        record, manifest,
        runner=lambda *args, **kwargs: Proc(stdout=stdout),
    )

    tool_path = (Path(__file__).resolve().parent.parent
                 / "tools" / "ham10000_cnn.py").resolve()
    executed = shlex.join([
        sys.executable, str(tool_path), "--image", record["image_path"]
    ])
    image_sha = hashlib.sha256(b"image").hexdigest()
    stdout_sha = hashlib.sha256(stdout.encode()).hexdigest()
    receipt_id = hashlib.sha256("\0".join((
        "ham10000-cnn", executed, image_sha, stdout_sha, "0"
    )).encode()).hexdigest()

    assert result.tool == "ham10000-cnn"
    assert result.exit_code == 0
    assert result.duration_ms is not None and result.duration_ms >= 0
    assert result.command_sha256 == hashlib.sha256(executed.encode()).hexdigest()
    assert result.image_sha256 == image_sha
    assert result.stdout_sha256 == stdout_sha
    assert result.receipt_id == receipt_id


def test_duplicate_claims_do_not_create_extra_votes(harness):
    record, manifest, command = harness
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return Proc(
            stdout=json.dumps({
                "status": "success",
                "disease_name": "Melanoma",
                "confidence_score": 0.8,
            })
        )

    claimed = claim(command).model_copy(update={"commands": [command, command]})
    rows = orchestrator.verify_receipts(
        claimed, record, manifest, runner=runner
    )

    assert len(calls) == 1
    assert len(rows) == 1

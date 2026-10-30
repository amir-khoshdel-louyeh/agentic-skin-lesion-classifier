"""Model/calibration artefact manifest tests."""

from pathlib import Path

import pytest

from control.artifacts import (
    ArtifactIntegrityError,
    DEFAULT_MANIFEST,
    build_manifest,
    enforce_manifest,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parent.parent


def test_artifact_manifest_detects_tampering_and_missing_files(tmp_path):
    (tmp_path / "calibration").mkdir()
    (tmp_path / "models" / "derm-cnn").mkdir(parents=True)
    (tmp_path / "calibration" / "demo.json").write_text('{"tool": "demo"}')
    (tmp_path / "calibration" / "SUMMARY.md").write_text("# Summary\n")
    model = tmp_path / "models" / "derm-cnn" / "model.pth"
    model.write_bytes(b"model-bytes")

    manifest_path = tmp_path / DEFAULT_MANIFEST
    manifest = write_manifest(manifest_path, tmp_path)
    assert manifest["schema_version"] == 1
    assert set(manifest["calibration"]) == {
        "calibration/demo.json", "calibration/SUMMARY.md"
    }
    assert set(manifest["models"]) == {"models/derm-cnn/model.pth"}

    clean = verify_manifest(tmp_path, manifest_path)
    assert clean["ok"] is True
    assert clean["checked"] == 3
    assert clean["problems"] == []

    calibration = tmp_path / "calibration" / "demo.json"
    original = calibration.read_text()
    calibration.write_text('{"tool": "tampered"}')
    tampered = verify_manifest(tmp_path, manifest_path)
    assert tampered["ok"] is False
    assert any("sha256 mismatch" in p for p in tampered["problems"])

    calibration.write_text(original)
    model.unlink()
    optional = verify_manifest(tmp_path, manifest_path)
    assert optional["ok"] is True
    assert any("missing model artifact" in w for w in optional["warnings"])
    strict = verify_manifest(
        tmp_path, manifest_path, allow_missing_models=False
    )
    assert strict["ok"] is False
    assert any("missing model artifact" in p for p in strict["problems"])

    model.write_bytes(b"model-bytes")
    (tmp_path / "calibration" / "unlisted.json").write_text("{}")
    unlisted = verify_manifest(tmp_path, manifest_path)
    assert unlisted["ok"] is False
    assert any("unlisted calibration" in p for p in unlisted["problems"])

    (tmp_path / "calibration" / "unlisted.json").unlink()
    calibration.unlink()
    missing = verify_manifest(tmp_path, manifest_path)
    assert missing["ok"] is False
    assert any("missing calibration artifact" in p for p in missing["problems"])


def test_runtime_gate_blocks_tampering_and_strict_missing_model(tmp_path):
    (tmp_path / "calibration").mkdir()
    (tmp_path / "models" / "derm-cnn").mkdir(parents=True)
    calibration = tmp_path / "calibration" / "demo.json"
    calibration.write_text('{"tool": "demo"}')
    model = tmp_path / "models" / "derm-cnn" / "model.pth"
    model.write_bytes(b"model-bytes")
    manifest_path = tmp_path / DEFAULT_MANIFEST
    write_manifest(manifest_path, tmp_path)

    accepted = enforce_manifest(tmp_path, manifest_path, strict_models=True)
    assert accepted["ok"] is True

    calibration.write_text('{"tool": "tampered"}')
    with pytest.raises(ArtifactIntegrityError, match="sha256 mismatch"):
        enforce_manifest(tmp_path, manifest_path, strict_models=False)

    calibration.write_text('{"tool": "demo"}')
    model.unlink()
    optional = enforce_manifest(tmp_path, manifest_path, strict_models=False)
    assert optional["ok"] is True
    assert optional["warnings"]
    with pytest.raises(ArtifactIntegrityError, match="missing model artifact"):
        enforce_manifest(tmp_path, manifest_path, strict_models=True)


def test_committed_manifest_covers_current_calibration():
    result = verify_manifest(
        ROOT, ROOT / DEFAULT_MANIFEST, allow_missing_models=True
    )
    assert result["problems"] == [], result
    assert result["checked"] >= len(list((ROOT / "calibration").glob("*")))

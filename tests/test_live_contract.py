"""Opt-in live contract smoke test.

This test runs a real diagnostic tool and therefore needs the local
checkpoint/runtime. It is excluded from the default fast tier and skips
unless RUN_LIVE_TESTS=1, so CI never starts a model by accident.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = (
    "status",
    "tool",
    "model_tier",
    "model_executed",
    "predicted_class_index",
    "disease_name",
    "confidence_score",
)


@pytest.mark.live
def test_live_diagnostic_contract():
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("set RUN_LIVE_TESTS=1 to run the live model smoke test")

    from control.manifest import load_manifest

    manifest = load_manifest(ROOT / "tools" / "manifest.yaml")
    name = os.environ.get("LIVE_CONTRACT_TOOL", "ham10000-cnn")
    tool = manifest.by_name(name)
    assert tool.status == "ready", f"{name} is not ready"

    record = yaml.safe_load((ROOT / "prompt.yaml").read_text(encoding="utf-8"))[0]
    command = [
        sys.executable,
        str(ROOT / tool.command),
        "--image",
        str(ROOT / record["image_path"]),
    ]
    if "metadata" in tool.requires:
        command += ["--metadata", json.dumps(record.get("metadata", {}))]

    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=900, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    payload = json.loads(proc.stdout)
    assert payload.get("status") == "success", payload
    for key in REQUIRED:
        assert key in payload, f"missing live contract key: {key}"

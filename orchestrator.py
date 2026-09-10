"""Orchestrator round loop (Phase 3, proposal.tmp Sections 4/8).

One record at a time: sequential isolated agents, receipt-only voting,
one-way escalation, Markdown report + JSONL audit. No diagnostic tools
here — control only.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.runner import run_agent
from control.decide import append_audit, build_report, decide
from control.manifest import load_manifest, render_for_agents
from control.paths import resolve, startup_path_check
from control.schemas import VerdictEnvelope

AGENT_ROLES = (
    "Triage role: use ONLY tools/ham10000_cnn.py with the case image. "
    "Report its exact JSON output faithfully.",
    "Fusion role: use ONLY tools/multimodal_fusion.py with the case image "
    "and the case metadata JSON. Report its exact JSON output faithfully.",
)


def load_record(prompt_file: Path, index: int) -> dict:
    records = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
    if index < 0 or index >= len(records):
        raise IndexError("record_index out of range")
    return records[index]


def verify_receipt(
    claimed: VerdictEnvelope,
    record: dict,
    manifest,  # control.manifest.Manifest
    timeout: int = 300,
) -> VerdictEnvelope:
    """Re-execute the claimed command control-side. Self-reported receipts
    are NOT trusted: mismatch => ran=False (no evidence). Only commands
    of the form `python tools/<ready-tool>.py --image <record image>
    [--metadata ...]` are eligible; anything else is rejected outright."""
    import json as _json
    import shlex
    import subprocess

    def rejected(reason: str) -> VerdictEnvelope:
        return VerdictEnvelope(
            ran=False,
            command=claimed.command,
            reasoning=f"Receipt rejected: {reason}",
            uncertainty_flags=["tool_failed"],
        )

    def corrected(reason: str, actual: dict) -> VerdictEnvelope:
        flags = list(dict.fromkeys(
            list(claimed.uncertainty_flags) + ["disagreement"]))
        return VerdictEnvelope(
            ran=True,
            command=claimed.command,
            exit_code=0,
            predicted_class=actual.get("disease_name"),
            confidence=(None if actual.get("confidence_score") is None
                        else float(actual["confidence_score"])),
            reasoning=f"{claimed.reasoning} [control: {reason}]",
            uncertainty_flags=flags,
        )

    if not claimed.ran or claimed.exit_code != 0:
        return claimed
    try:
        parts = shlex.split(claimed.command)
    except ValueError:
        return rejected("unparseable command")
    if len(parts) < 4 or parts[0] != "python" or parts[2] != "--image":
        return rejected("command not in verifiable form")
    tool_rel, image_arg = parts[1], parts[3]
    try:
        tool = next(
            t for t in manifest.tools
            if t.status == "ready" and t.command == tool_rel
        )
    except StopIteration:
        return rejected(f"unknown or non-ready tool: {tool_rel}")
    record_image = str(resolve(str(record.get("image_path"))))
    if str(resolve(image_arg)) != record_image:
        return rejected("image path does not match the record")
    cmd = [sys.executable, str(resolve(tool_rel)), "--image", record_image]
    if "--metadata" in parts:
        # Metadata is re-taken from the record, never from agent text:
        # agents routinely emit unquoted/broken JSON here.
        cmd += ["--metadata", _json.dumps(record.get("metadata", {}))]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(ROOT_DIR),
        )
    except Exception as exc:
        return rejected(f"re-execution failed: {exc}")
    if proc.returncode != 0:
        return rejected(f"re-execution exit {proc.returncode}: "
                        f"{proc.stderr.strip()[-300:]}")
    try:
        actual = _json.loads(proc.stdout)
    except (ValueError, TypeError):
        return rejected("re-execution produced no JSON")
    if actual.get("status") != "success":
        return rejected(f"tool reported: {actual.get('message')}")
    if actual.get("disease_name") != claimed.predicted_class:
        return corrected(
            f"class corrected (agent said {claimed.predicted_class})",
            actual)
    actual_conf = actual.get("confidence_score")
    if (claimed.confidence is not None and actual_conf is not None
            and abs(float(actual_conf) - float(claimed.confidence)) > 1e-3):
        return corrected(
            f"confidence corrected (agent said {claimed.confidence})",
            actual)
    return claimed


def run_round(
    record_index: int,
    prompt_file: str = "prompt.yaml",
    tool_helper_file: str = "tools/manifest.yaml",
    agent_id: str = "main",
) -> Path:
    startup_path_check()
    prompt_path = resolve(prompt_file)
    record = load_record(prompt_path, record_index)
    manifest = load_manifest(resolve(tool_helper_file))
    manifest_text = render_for_agents(manifest)

    verdicts = []
    for extra in AGENT_ROLES:
        record_with_role = dict(record)
        record_with_role["role_extra"] = extra
        claimed = run_agent(record_with_role, manifest_text, agent_id=agent_id)
        verdicts.append(verify_receipt(claimed, record, manifest))

    decision = decide(verdicts)
    report = build_report(record, verdicts, decision)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT_DIR / "reports" / f"round_{record_index}_{stamp}.md"
    out.write_text(report, encoding="utf-8")
    append_audit({
        "event": "round",
        "record_index": record_index,
        "image": record.get("image_path"),
        "decision": decision["decision"],
        "reason": decision["reason"],
        "report": out.name,
    })
    return out

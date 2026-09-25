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
from control.guard import run_guard
from control.manifest import load_manifest, render_for_agents
from control.paths import resolve, startup_path_check
from control.prompts import effective_thresholds
from control.router import route, run_abcde
from control.prompts import effective_thresholds
from control.schemas import VerdictEnvelope

AGENT_ROLES = (
    "Triage role: use ONLY tools/ham10000_cnn.py with the case image. "
    "Report its exact JSON output faithfully.",
    "Fusion role: use ONLY tools/multimodal_fusion.py with the case image "
    "and the case metadata JSON. Report its exact JSON output faithfully.",
    "High-tier role: use ONLY tools/ensemble_high.py with the case image "
    "and the case metadata JSON. Report its exact JSON output faithfully, "
    "including entropy and uncertainty_flags.",
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
    borderline_threshold: float | None = None,
) -> Path:
    startup_path_check()
    # Physician customization (Phase 8): the doctor's validated thresholds
    # drive the decision — never a hardcoded constant. Invalid doctor files
    # fail the round fast instead of running with silent defaults.
    thresholds = effective_thresholds("orchestrator")
    if borderline_threshold is None:
        borderline_threshold = thresholds["borderline_threshold"]
    prompt_path = resolve(prompt_file)
    record = load_record(prompt_path, record_index)
    manifest = load_manifest(resolve(tool_helper_file))
    manifest_text = render_for_agents(manifest)

    # Phase 10 routing: deterministic evidence first. The guard never
    # stops a round (reference images fail its blur bar); it only forces
    # the full path with reasons. "screen" runs the triage worker alone.
    guard = run_guard(str(record.get("image_path")))
    abcde = run_abcde(str(record.get("image_path")))
    route_info = route(guard, abcde)
    roles = AGENT_ROLES if route_info["path"] == "full" else AGENT_ROLES[:1]

    verdicts = []
    for extra in roles:
        record_with_role = dict(record)
        record_with_role["role_extra"] = extra
        claimed = run_agent(record_with_role, manifest_text, agent_id=agent_id)
        verdicts.append(verify_receipt(claimed, record, manifest))

    decision = decide(verdicts, borderline_threshold=borderline_threshold)
    report = build_report(record, verdicts, decision, thresholds=thresholds)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT_DIR / "report" / f"round_{record_index}_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    append_audit({
        "event": "round",
        "record_index": record_index,
        "image": record.get("image_path"),
        "route": route_info,
        "guard": {"passed": guard["passed"], "flags": guard["flags"]},
        "abcde": {"risk_band": abcde["risk_band"],
                  "score": abcde["score"]},
        "decision": decision["decision"],
        "reason": decision["reason"],
        "report": out.name,
        "thresholds": thresholds,
    })
    ping(f"Round {record_index} complete: {decision['decision']} ({out.name})",
         level="info")
    return out


def ping(message: str, level: str = "info") -> None:
    """Best-effort desktop toast (Phase 6 notify hook). A missing desktop
    or failing hook must never break a round: all errors swallowed."""
    import subprocess

    try:
        subprocess.run(
            [sys.executable, str(ROOT_DIR / "tools" / "notify.py"),
             "--message", message, "--level", level],
            capture_output=True, timeout=30, cwd=str(ROOT_DIR),
        )
    except Exception:  # noqa: BLE001 - notification is informational only
        pass

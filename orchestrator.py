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
from control.critic import run_critic
from control.decide import append_audit, build_report, decide
from control.specialists import shortlist, specialist_brief
from control.guard import run_guard
from control.manifest import load_manifest, render_for_agents
from control.paths import resolve, startup_path_check
from control.prompts import effective_thresholds
from control.router import run_abcde
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

    try:
        from control.vram import unload_ollama as _unload
        _unload()
    except Exception:
        pass

    def rejected(reason: str) -> VerdictEnvelope:
        return VerdictEnvelope(
            ran=False,
            command=claimed.command,
            reasoning=f"Receipt rejected: {reason}",
            uncertainty_flags=["tool_failed"],
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
    # (No class-mismatch branch: the single-writer return below always
    # carries the tool's own class; the agent's claim is only noted.)
    actual_conf = actual.get("confidence_score")
    # Single-writer principle: numbers come ONLY from re-execution, never
    # from agent text. The agent's numbers are kept as a reliability
    # signal (exact/rounded/fabricated), but the verdict always carries
    # the tool's own values — there is nothing left to fabricate.
    claimed_conf = claimed.confidence
    drift = (None if claimed_conf is None or actual_conf is None
             else abs(float(actual_conf) - float(claimed_conf)))
    if drift is None:
        reliability = "omitted"
    elif drift <= 0.01:
        reliability = "exact"
    elif drift <= 0.05:
        reliability = "rounded"
    else:
        reliability = "fabricated"
    flags = list(dict.fromkeys(
        list(claimed.uncertainty_flags)
        + (["disagreement"] if drift is not None and drift > 0.01 else [])))
    note = (f"{claimed.reasoning} [control: agent said "
            f"{claimed.predicted_class}/{claimed_conf}; "
            f"reliability={reliability}]")
    return VerdictEnvelope(
        ran=True,
        command=claimed.command,
        exit_code=0,
        predicted_class=actual.get("disease_name"),
        confidence=(None if actual_conf is None
                    else float(actual_conf)),
        reasoning=note,
        uncertainty_flags=[f for f in flags
                           if f in ("borderline", "no_evidence",
                                    "tool_failed", "disagreement")],
    )


def run_ensemble_verdict(record: dict, timeout: int = 600,
                         ) -> VerdictEnvelope | None:
    """Control-side ensemble vote for careful rounds: runs the high-tier
    tool directly (no LLM) and wraps its JSON as a receipted verdict.
    Returns None when the tool fails — never a guess."""
    import json as _json
    import subprocess as _subprocess

    try:
        from control.vram import unload_ollama as _unload3
        _unload3()
    except Exception:
        pass

    cmd = [sys.executable, str(resolve("tools/ensemble_high.py")),
           "--image", str(resolve(str(record.get("image_path")))),
           "--metadata", _json.dumps(record.get("metadata", {}))]
    try:
        proc = _subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=str(ROOT_DIR))
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = _json.loads(proc.stdout)
    except (ValueError, TypeError):
        return None
    if payload.get("status") != "success":
        return None
    try:
        return VerdictEnvelope(
            ran=True,
            command=" ".join(cmd),
            exit_code=0,
            predicted_class=payload.get("disease_name"),
            confidence=(None if payload.get("confidence_score") is None
                        else float(payload["confidence_score"])),
            reasoning="control-side ensemble vote (careful route)",
            uncertainty_flags=[
                f for f in (payload.get("uncertainty_flags") or [])
                if f in ("borderline", "no_evidence", "tool_failed",
                         "disagreement")],
        )
    except Exception:
        return None


def run_round(
    record_index: int,
    prompt_file: str = "prompt.yaml",
    tool_helper_file: str = "tools/manifest.yaml",
    agent_id: str = "main",
    borderline_threshold: float | None = None,
    careful: bool = False,
) -> Path:
    startup_path_check()
    # Physician customization (Phase 8): the doctor's validated thresholds
    # drive the decision — never a hardcoded constant. Invalid doctor files
    # fail the round fast instead of running with silent defaults.
    thresholds = effective_thresholds("orchestrator")
    if borderline_threshold is None:
        borderline_threshold = thresholds["borderline_threshold"]
    if careful:
        # "Be more careful" is a concrete route, not effort: one step
        # tighter within doctor bounds, logged in the audit.
        borderline_threshold = min(borderline_threshold + 0.05, 0.9)
    prompt_path = resolve(prompt_file)
    record = load_record(prompt_path, record_index)
    manifest = load_manifest(resolve(tool_helper_file))
    manifest_text = render_for_agents(manifest)

    import time as _time
    timings: dict[str, float] = {}
    # plan.tmp T1: deterministic routing RETIRED. No route()/screen shortcut.
    # Guard + ABCDE remain as plain evidence (for critic flags + audit);
    # they never decide a path. Every round runs the full worker set;
    # the fast path will re-emerge from Brain early-stop (T3), not code.
    # NOTE (T2): these control-side calls move into the Quality Agent
    # (chain step 1); kept here only until T2 wires the agent.
    _t0 = _time.perf_counter()
    guard = run_guard(str(record.get("image_path")))
    timings["guard_s"] = round(_time.perf_counter() - _t0, 2)
    _t0 = _time.perf_counter()
    abcde = run_abcde(str(record.get("image_path")))
    timings["abcde_s"] = round(_time.perf_counter() - _t0, 2)
    route_info = {"path": "full",
                  "reasons": ["deterministic routing retired (T1); "
                              "full path always"],
                  "retired": True}
    preprocess_info: dict | None = None
    verdicts = []
    llm_sessions = 0
    _t0 = _time.perf_counter()
    for extra in AGENT_ROLES:
        record_with_role = dict(record)
        record_with_role["role_extra"] = extra
        claimed = run_agent(record_with_role, manifest_text, agent_id=agent_id)
        llm_sessions += 1
        verdicts.append(verify_receipt(claimed, record, manifest))
    timings["workers_s"] = round(_time.perf_counter() - _t0, 2)

    _t0 = _time.perf_counter()
    if careful:
        extra = run_ensemble_verdict(record)
        if extra is not None:
            verdicts.append(extra)
    timings["careful_ensemble_s"] = round(_time.perf_counter() - _t0, 2)

    # Critic runs once per round, after receipts: one advisory session
    # that can only sustain an objection, never confirm. (T1: always runs;
    # the old `if path == full` gate is gone with routing.)
    critic_sustained, critic_reasons = False, []
    _t0 = _time.perf_counter()
    critic_sustained, critic_reasons = run_critic(
        verdicts,
        list(guard.get("flags", []))
        + ([f"abcde:{abcde['risk_band']}"] if abcde.get("risk_band")
           else []),
        agent_id=agent_id,
    )
    timings["critic_s"] = round(_time.perf_counter() - _t0, 2)

    # Specialists run on disagreement only: one advocate per shortlisted
    # disease (max 2, sequential), same envelope schema and receipt
    # verification as every other worker.
    _t0 = _time.perf_counter()
    diseases = shortlist(verdicts) or []
    specialist_verdicts = []
    for disease in diseases:
        record_with_role = dict(record)
        record_with_role["role_extra"] = specialist_brief(disease,
                                                          diseases)
        claimed = run_agent(record_with_role, manifest_text,
                            agent_id=agent_id)
        verified = verify_receipt(claimed, record, manifest)
        # Phase 11 acceptance: off-shortlist classes never count. The tool
        # output is faithful, but a specialist confined to `disease` may not
        # introduce a third candidate — it becomes no-evidence.
        if verified.is_countable() and verified.predicted_class != disease:
            verified = VerdictEnvelope(
                ran=False,
                command=verified.command,
                reasoning=(f"Specialist off-shortlist ({verified.predicted_class} "
                           f"not in {diseases}) — recorded as no evidence."),
                uncertainty_flags=["tool_failed"],
            )
        specialist_verdicts.append(verified)
    timings["specialists_s"] = round(_time.perf_counter() - _t0, 2)
    verdicts.extend(specialist_verdicts)

    decision = decide(verdicts, borderline_threshold=borderline_threshold,
                      critic_objection=critic_sustained)
    # Persistent specialist disagreement is named, never averaged away:
    # two confident advocates for different diseases must refer.
    spec_classes = sorted({
        v.predicted_class for v in specialist_verdicts if v.is_countable()
    })
    if len(spec_classes) > 1:
        decision = {
            "decision": "suspicious",
            "reason": (f"specialist dilemma: {spec_classes[0]} vs "
                       f"{spec_classes[1]}"),
            "verdicts_counted": sum(1 for v in verdicts if v.is_countable()),
        }
    report = build_report(record, verdicts, decision, thresholds=thresholds)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT_DIR / "report" / f"round_{record_index}_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    append_audit({
        "event": "round",
        "record_index": record_index,
        "image": record.get("image_path"),
        "careful": careful,
        "route": route_info,
        "llm_sessions": llm_sessions + 1 + len(specialist_verdicts),
        "timings": timings,
        "guard": {"passed": guard["passed"], "flags": guard["flags"]},
        "abcde": {"risk_band": abcde["risk_band"],
                  "score": abcde["score"]},
        "preprocess": preprocess_info or {},
        "critic": {"sustained": critic_sustained,
                   "reasons": critic_reasons},
        "specialists": {"shortlist": diseases,
                        "counted": spec_classes},
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

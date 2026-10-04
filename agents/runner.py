"""Agent runner: isolated OpenClaw sessions per diagnostic thinker.

Each session gets: role prompt (system + doctor layers) + case + Quality
text + Brain steer + full diagnostic pool + JSON instruction. Sessions
are isolated, then destroyed; they never see each other directly, only
via the steer note. Output goes through parse/validate with exactly one
repair retry; double failure becomes a ran=False envelope so the
orchestrator records "no evidence" instead of guessing.
"""

import json
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.enforcement import (
    JSON_INSTRUCTION,
    EnvelopeError,
    build_repair_prompt,
    parse_envelope,
)
from control.prompts import load_role
from control.schemas import VerdictEnvelope
from skin_agent import run_openclaw_cli


def _release_llm() -> None:
    """Best-effort LLM unload after a session (sequential GPU)."""
    try:
        from control.vram import unload_ollama as _unload
        _unload()
    except Exception:
        pass


def build_agent_prompt(
    record: dict,
    manifest_text: str,
    role: str = "agent",
    quality_text: str = "",
    steer: str = "",
    hint: tuple | list = (),
) -> str:
    role_prompt = load_role(role)
    metadata = json.dumps(record.get("metadata", {}), ensure_ascii=False)
    extra = record.get("role_extra", "")
    extra_section = f"\nRole assignment: {extra}\n" if extra else ""
    quality_section = (f"\nQuality Agent (chain step 1): {quality_text}\n"
                       if quality_text else "")
    command_form = (
        "\nCommand form (no exceptions):\n"
        "- Every command MUST start with the word `python`.\n"
        "- Name the exact tools/*.py file from the pool above "
        "(underscores, e.g. ham10000_cnn.py) — a skill name with dashes "
        "is NOT a command and is rejected as no evidence.\n"
        "- Example: python tools/ham10000_cnn.py --image <path_to_image>\n"
    )
    exact_choice = (
        "\nFree choice, zero transcription:\n"
        "- FIRST run at least one tool command, THEN write the envelope. "
        "Intent without execution counts as no evidence — narrating what "
        "you WOULD run earns nothing.\n"
        "- ran=false is ONLY for a tool that crashed: include the exact "
        "command and its error output in reasoning.\n"
        "- Choose 1..n LINES above and run each EXACTLY as written "
        "(copy-paste the whole line). Do not retype, rename, or shorten.\n"
        "- Use ONLY the case image path inside those lines. There is NO "
        "cleaned-image prerequisite: if no cleaned_image path was given, "
        "use the case image. Never refuse for a missing cleaned image.\n"
        "- Quality flags NEVER block tool execution — run regardless of "
        "the Quality verdict.\n"
        "- In your envelope, set predicted_class and confidence to null "
        "ALWAYS. NEVER write tool numbers from memory — control fills "
        "them from re-execution. Explain WHY you chose these tools, not "
        "their numeric results.\n"
    )
    hint_names = ", ".join(hint) if hint else "none"
    steer_section = (
        f"\nBrain steer for THIS session only (previous sessions stay "
        f"invisible to you): {steer}\n"
        f"Suggested subset is a HINT only, you may ignore it: "
        f"{hint_names}\n"
        if steer else ""
    )
    return (
        f"{role_prompt}\n{extra_section}{quality_section}\n"
        f"Case image: {record.get('image_path')}\n"
        f"Case metadata: {metadata}\n"
        f"Available tools (FULL pool, free choice):\n{manifest_text}\n"
        f"{command_form}"
        f"{exact_choice}"
        f"{steer_section}\n"
        "Decide yourself which 1..n tools to run, in which order, then "
        "report.\n" + JSON_INSTRUCTION
    )


def extract_text(response: dict) -> str:
    if isinstance(response, dict) and response.get("payloads"):
        return "\n".join(item.get("text", "") for item in response["payloads"])
    return json.dumps(response, ensure_ascii=False)


def run_agent(
    record: dict,
    manifest_text: str,
    agent_id: str = "main",
    quality_text: str = "",
    steer: str = "",
    hint: tuple | list = (),
) -> VerdictEnvelope:
    session = f"agent-{uuid.uuid4().hex}"
    prompt = build_agent_prompt(record, manifest_text,
                                quality_text=quality_text,
                                steer=steer, hint=hint)
    try:
        raw = extract_text(run_openclaw_cli(prompt, agent_id=agent_id))
        _release_llm()
    except Exception as exc:
        return VerdictEnvelope(
            ran=False,
            command="(orchestration failed)",
            reasoning=f"OpenClaw call failed: {exc}",
            uncertainty_flags=["tool_failed"],
        )
    try:
        return parse_envelope(raw)
    except EnvelopeError as first_error:
        try:
            retry_raw = extract_text(
                run_openclaw_cli(
                    build_repair_prompt(raw, str(first_error)),
                    agent_id=agent_id,
                )
            )
            _release_llm()
            return parse_envelope(retry_raw)
        except (EnvelopeError, Exception) as exc:
            return VerdictEnvelope(
                ran=False,
                command="(agent output rejected)",
                reasoning=f"Invalid envelope after repair retry: {exc}",
                uncertainty_flags=["tool_failed"],
            )


RECEIPT_REJECTED = "Receipt rejected:"


def build_receipt_repair_prompt(bad_command: str, reason: str,
                                manifest_text: str, quality_text: str = "",
                                steer: str = "") -> str:
    return (
        "Your reported command was REJECTED and counted as no evidence:\n"
        f"command: {bad_command}\nreason: {reason}\n\n"
        "Do not explain. Reply ONCE with a corrected single JSON verdict "
        "envelope (same schema), using one of these EXACT lines "
        "(copy-paste the whole line):\n"
        f"{manifest_text}\n"
        + (f"\nQuality: {quality_text}\n" if quality_text else "")
        + (f"\nActive steer: {steer}\n" if steer else "")
        + JSON_INSTRUCTION
    )


def run_session(
    record: dict,
    manifest_text: str,
    verify,
    agent_id: str = "main",
    quality_text: str = "",
    steer: str = "",
    hint: tuple | list = (),
) -> VerdictEnvelope:
    """Claim → verify → one targeted repair on receipt rejection.

    `verify` is a callable taking a VerdictEnvelope and returning the
    re-executed VerdictEnvelope (e.g. a verify_receipt partial). Genuine
    ran=false passes through untouched — only a *claimed* run whose
    command the referee rejects gets exactly one correction retry.
    Never raises: verification errors become no-evidence.
    """
    claimed = run_agent(record, manifest_text, agent_id=agent_id,
                        quality_text=quality_text, steer=steer, hint=hint)
    try:
        verified = verify(claimed)
    except Exception as exc:  # noqa: BLE001 - referee must not crash
        return VerdictEnvelope(
            ran=False,
            command=claimed.command,
            reasoning=f"Verification failed: {exc}",
            uncertainty_flags=["tool_failed"],
        )
    if not (claimed.ran and not verified.ran
            and verified.reasoning.startswith(RECEIPT_REJECTED)):
        return verified
    try:
        repair_raw = extract_text(run_openclaw_cli(
            build_receipt_repair_prompt(claimed.command,
                                        verified.reasoning,
                                        manifest_text, quality_text,
                                        steer),
            agent_id=agent_id,
        ))
        _release_llm()
    except Exception:  # noqa: BLE001 - keep the first rejection
        return verified
    try:
        corrected = parse_envelope(repair_raw)
    except EnvelopeError:
        return verified
    try:
        return verify(corrected)
    except Exception:  # noqa: BLE001 - keep the first rejection
        return verified

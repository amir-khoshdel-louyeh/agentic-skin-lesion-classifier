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
            return parse_envelope(retry_raw)
        except (EnvelopeError, Exception) as exc:
            return VerdictEnvelope(
                ran=False,
                command="(agent output rejected)",
                reasoning=f"Invalid envelope after repair retry: {exc}",
                uncertainty_flags=["tool_failed"],
            )

"""Agent runner: isolated OpenClaw sessions per diagnostic agent.

Each agent gets: role prompt (system + doctor layers) + record + tool
manifest + JSON instruction. Output goes through parse/validate with
exactly one repair retry; double failure becomes a ran=False envelope
so the orchestrator records "no evidence" instead of guessing.
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
) -> str:
    role_prompt = load_role(role)
    metadata = json.dumps(record.get("metadata", {}), ensure_ascii=False)
    extra = record.get("role_extra", "")
    extra_section = f"\nRole assignment: {extra}\n" if extra else ""
    return (
        f"{role_prompt}\n{extra_section}\nCase image: {record.get('image_path')}\n"
        f"Case metadata: {metadata}\n"
        f"Available tools:\n{manifest_text}\n\n"
        "Execute the suitable tool command(s) for this case, then "
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
) -> VerdictEnvelope:
    session = f"agent-{uuid.uuid4().hex}"
    prompt = build_agent_prompt(record, manifest_text)
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

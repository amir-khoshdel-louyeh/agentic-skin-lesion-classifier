"""OpenClaw CLI bridge and prompt-record loading.

Shared transport for every chain role: Quality (chain step 1), the
Brain-loop thinker sessions (§2.3), the Critic (step 3), the Specialists
(step 4), and the post-lock Q&A sidecar (§2.7). Each call runs one
isolated OpenClaw session (its own session id) and returns the parsed
JSON payload.

plan.tmp §6 retires the old prompt-only pipeline (one mega-prompt, no
receipts, no chain). Rounds now run through `main.py` /
`orchestrator.run_round`; this module only carries text to and from the
local OpenClaw agent.
"""

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_prompt_records(prompt_file: Path) -> List[Dict[str, Any]]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        try:
            records = yaml.safe_load(f)
            if not isinstance(records, list):
                raise ValueError("YAML content must be a list of records.")
            return records
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML format: {exc}")


def find_openclaw_executable() -> str:
    command = shutil.which("openclaw") or shutil.which("openclaw.cmd")
    if not command:
        raise RuntimeError(
            "OpenClaw CLI not found in PATH. Ensure `openclaw` is installed "
            "and available from your shell."
        )
    return command


def is_transient_openclaw_error(output: str) -> bool:
    lower = output.lower()
    return any(
        marker in lower
        for marker in [
            "embeddedattemptsessiontakeovererror",
            "session file changed while embedded prompt lock was released",
            "auto-compaction",
            "willretry=false",
            "threshold",
            "incomplete turn detected",
            "failovererror",
            "candidate_failed",
            "incomplete terminal response",
        ]
    )


def run_openclaw_cli(
    prompt: str, agent_id: str = "main", show_command: bool = True
) -> Dict[str, Any]:
    openclaw_cmd = find_openclaw_executable()
    clean_prompt = " ".join(prompt.strip().split())

    current_session_id = f"session-{uuid.uuid4().hex}"

    max_retries = 2
    for attempt in range(max_retries + 1):
        command = [
            openclaw_cmd,
            "agent",
            "--agent",
            agent_id,
            "--local",
            "--session-id",
            current_session_id,
            f"--message={clean_prompt}",
            "--json",
            "--thinking",
            "off",
        ]

        if show_command:
            print("Command:", " ".join(command))

        try:
            import os as _os

            _env = dict(_os.environ)
            # Sequential GPU schedule (§5): do not keep the agent LLM
            # resident while PyTorch tools load. Best-effort only.
            _env.setdefault("OLLAMA_KEEP_ALIVE", "0")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1200,
                env=_env,
            )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "OpenClaw CLI timed out after 1200 seconds. "
                "Ensure the OpenClaw gateway and local tools are running."
            ) from exc

        if result.returncode != 0:
            stderr_text = result.stderr.strip()
            stdout_text = result.stdout.strip()
            combined = f"{stderr_text}\n{stdout_text}".strip()

            if attempt < max_retries and is_transient_openclaw_error(combined):
                print(
                    f"Transient OpenClaw error detected on attempt "
                    f"{attempt + 1}: {combined}. "
                    "Retrying with a fresh session..."
                )
                time.sleep(2)
                continue

            raise RuntimeError(
                "OpenClaw CLI failed: "
                + (stderr_text or stdout_text or "unknown error")
            )

        try:
            return json.loads(result.stdout)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenClaw CLI returned invalid JSON:\n{result.stdout}"
            ) from exc

    raise RuntimeError("OpenClaw CLI failed after all retry attempts.")

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
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml

from control.inference_guard import InferencePolicy, InferenceTimeout


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
            "incomplete turn detected",
            "failovererror",
            "candidate_failed",
            "incomplete terminal response",
            # Rate limiting, but matched specifically: a bare "429" or
            # "threshold" used to retry ANY error mentioning those digits
            # or that word, five times with backoff, on a permanent fault.
            "http 429",
            "status 429",
            "error 429",
            "429 too many",
            "429 rate limit",
            "rate_limit_exceeded",
            "too_many_requests",
            "resource_exhausted",
            "rate limit",
        ]
    )


def run_openclaw_cli(
    prompt: str,
    agent_id: str = "main",
    show_command: bool = True,
    policy: InferencePolicy | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    openclaw_cmd = find_openclaw_executable()
    clean_prompt = " ".join(prompt.strip().split())
    model = model or os.environ.get("SKIN_AGENT_MODEL") or None
    policy = policy or InferencePolicy.from_env(
        "OPENCLAW", default_timeout=600.0,
        default_attempts=3,
    )
    deadline = time.monotonic() + policy.max_total_s

    for attempt in range(policy.max_attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InferenceTimeout(
                f"OpenClaw inference budget of {policy.max_total_s:g}s exhausted"
            )
        # A fresh session per attempt is the whole point of the retry: the
        # errors it targets are session-state errors, so reusing one id
        # would reproduce the failure it is meant to escape.
        current_session_id = f"session-{uuid.uuid4().hex}"
        command = [
            openclaw_cmd,
            "agent",
            "--agent",
            agent_id,
        ]
        if model:
            command += ["--model", model]
        command += [
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
            _env = dict(os.environ)
            # Sequential GPU schedule (§5): do not keep the agent LLM
            # resident while PyTorch tools load. Best-effort only.
            _env.setdefault("OLLAMA_KEEP_ALIVE", "0")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(policy.timeout_s, remaining),
                env=_env,
            )

        except subprocess.TimeoutExpired as exc:
            raise InferenceTimeout(
                "OpenClaw CLI timed out after "
                f"{min(policy.timeout_s, remaining):g} seconds. "
                "Ensure the OpenClaw gateway and local tools are running."
            ) from exc

        if result.returncode != 0:
            stderr_text = result.stderr.strip()
            stdout_text = result.stdout.strip()
            combined = f"{stderr_text}\n{stdout_text}".strip()

            if (attempt + 1 < policy.max_attempts
                    and is_transient_openclaw_error(combined)):
                delay = policy.delay_for(attempt)
                if time.monotonic() + delay >= deadline:
                    raise InferenceTimeout(
                        "no OpenClaw retry budget remains after "
                        f"{attempt + 1} attempts"
                    )
                print(
                    f"Transient OpenClaw error detected on attempt "
                    f"{attempt + 1}: {combined}. "
                    f"Retrying in {delay}s with a fresh session..."
                )
                time.sleep(delay)
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

    raise RuntimeError(
        f"OpenClaw CLI failed after {policy.max_attempts} attempts."
    )

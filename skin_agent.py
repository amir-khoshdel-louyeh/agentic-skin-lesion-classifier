import argparse
import json
import shutil
import subprocess
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PROMPT_FILE = Path("prompt.txt")
DEFAULT_TOOL_HELP_FILE = Path("tool_manifest.md")
DEFAULT_OPENCLAW_PROMPT = textwrap.dedent(
    """
    You are an OpenClaw orchestrator for skin lesion analysis.
    A new record has arrived containing an image path, optional metadata, and a clinical prompt.
    Use the local CLI scripts listed in the helper manifest.
    Execute the exact command shown in the helper manifest, then return a final concise Markdown clinical report.
    Do not produce planning text, progress updates, or still running language.
    Do not include any intermediate monitoring commentary.
    Do not say the tool is running or that you will wait; instead, execute the tool immediately and respond only once it has finished.
    If the first tool returns low confidence, automatically escalate to the more accurate tool.
    """
)


def load_prompt_records(prompt_file: Path) -> List[Dict[str, Any]]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    records: List[Dict[str, Any]] = []
    for raw_line in prompt_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record in {prompt_file}: {exc}\nLine: {line}") from exc

        if "image_path" not in record or "prompt" not in record:
            raise ValueError(
                "Each record must contain at least 'image_path' and 'prompt' fields."
            )

        records.append(record)

    if not records:
        raise ValueError(f"No valid prompt records found in {prompt_file}")

    return records


def load_tool_helper(tool_helper_file: Path) -> str:
    if not tool_helper_file.exists():
        return ""
    return tool_helper_file.read_text(encoding="utf-8").strip()


def build_prompt(record: Dict[str, Any], tool_helper_file: Optional[Path] = None) -> str:
    metadata = record.get("metadata")
    metadata_section = ""
    if metadata is not None:
        pretty_metadata = json.dumps(metadata, indent=2, ensure_ascii=False)
        metadata_section = f"\nMetadata:\n{pretty_metadata}\n"

    image_path = record["image_path"]
    prompt_text = record["prompt"]

    helper_section = ""
    if tool_helper_file is not None:
        helper_text = load_tool_helper(tool_helper_file)
        if helper_text:
            helper_section = f"\nAvailable local tools:\n{helper_text}\n"

    return textwrap.dedent(
        f"""
        {DEFAULT_OPENCLAW_PROMPT}

        Image path: {image_path}
        {metadata_section}
        User prompt: {prompt_text}
        {helper_section}
        Use the available local tools and return a final report in Markdown.
        """
    )


def find_openclaw_executable() -> str:
    command = shutil.which("openclaw") or shutil.which("openclaw.cmd")
    if not command:
        raise RuntimeError(
            "OpenClaw CLI not found in PATH. Ensure `openclaw` is installed and available from your shell."
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


def run_openclaw_cli(prompt: str, agent_id: str = "main") -> Dict[str, Any]:
    session_id = f"prompt-{uuid.uuid4().hex}"
    openclaw_cmd = find_openclaw_executable()
    clean_prompt = " ".join(prompt.strip().split())
    command = [
        openclaw_cmd,
        "agent",
        "--agent",
        agent_id,
        "--local",
        "--session-id",
        session_id,
        f"--message={clean_prompt}",
        "--json",
        "--thinking",
        "off",
    ]

    print("Running OpenClaw agent command. This may take a while if the tool needs to load models.")
    print("Command:", " ".join(command))

    max_retries = 2
    for attempt in range(1, max_retries + 2):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "OpenClaw CLI timed out after 600 seconds. Ensure the OpenClaw gateway and local tools are running."
            ) from exc

        if result.returncode != 0:
            stderr_text = result.stderr.strip()
            stdout_text = result.stdout.strip()
            combined = f"{stderr_text}\n{stdout_text}".strip()
            if attempt <= max_retries and is_transient_openclaw_error(combined):
                print(f"Transient OpenClaw error detected on attempt {attempt}; retrying...")
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
                f"OpenClaw CLI returned invalid JSON: {result.stdout}"
            ) from exc


def response_is_incomplete(payload_text: str) -> bool:
    if not isinstance(payload_text, str):
        return True
    lower_text = payload_text.lower()
    incomplete_markers = [
        "still running",
        "please wait",
        "waiting",
        "in progress",
        "not yet",
        "monitor",
        "progress",
        "will provide",
        "will execute",
        "executing",
        "continue",
        "ongoing",
        "planning",
        "plan",
        "retry",
        "unexpected token",
        "suggests there might be issues",
        "should resolve any syntax issues",
        "this command should resolve",
        "for a more definitive diagnosis",
        "escalate to",
        "command should resolve",
        "there are no indications",
    ]
    return any(marker in lower_text for marker in incomplete_markers)


def build_correction_prompt(
    record: Dict[str, Any],
    previous_response: str,
    tool_helper_file: Optional[Path] = None,
) -> str:
    base_prompt = build_prompt(record, tool_helper_file=tool_helper_file)
    return textwrap.dedent(
        f"""
        {base_prompt}

        The previous OpenClaw response was not a valid final report. It used planning, debugging, or waiting language instead of returning a completed clinical result.
        Retry the task once and return only the final concise Markdown clinical report with the exact command that was executed.
        Do not say the tool is running, that you are waiting, or that you will continue monitoring.
        Do not explain shell quoting or syntax issues; instead, execute the tool command exactly and show the actual output.
        If the tool fails, print the raw tool error output and do not hallucinate a diagnosis.
        Use the same available local tools and escalate if needed.

        Previous response:
        {previous_response}
        """
    )


def send_records_to_openclaw(
    prompt_file: Path,
    record_index: Optional[int] = None,
    agent_id: str = "main",
    tool_helper_file: Optional[Path] = None,
) -> None:
    records = load_prompt_records(prompt_file)
    if record_index is not None:
        if record_index < 0 or record_index >= len(records):
            raise IndexError("record_index is out of range")
        records = [records[record_index]]

    for idx, record in enumerate(records):
        print("=" * 70)
        print(f"Processing prompt record {idx + 1}/{len(records)}")
        print(f"Image path: {record['image_path']}")

        prompt = build_prompt(record, tool_helper_file=tool_helper_file)

        response = run_openclaw_cli(prompt, agent_id=agent_id)
        if isinstance(response, dict) and response.get("payloads"):
            payload_text = "\n".join(
                item.get("text", "") for item in response["payloads"]
            )
        else:
            payload_text = json.dumps(response, indent=2, ensure_ascii=False)

        print("--- OpenClaw response ---")
        print(payload_text)

        if response_is_incomplete(payload_text):
            print("OpenClaw response appears incomplete or planning-oriented. Retrying with a correction prompt.")
            correction_prompt = build_correction_prompt(record, payload_text, tool_helper_file=tool_helper_file)
            retry_response = run_openclaw_cli(correction_prompt, agent_id=agent_id)
            if isinstance(retry_response, dict) and retry_response.get("payloads"):
                payload_text = "\n".join(
                    item.get("text", "") for item in retry_response["payloads"]
                )
            else:
                payload_text = json.dumps(retry_response, indent=2, ensure_ascii=False)

            print("--- OpenClaw correction response ---")
            print(payload_text)

        print("" + "=" * 70 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send prompt records from prompt.txt to OpenClaw one at a time."
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Path to the prompt file containing one JSON record per line.",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=None,
        help="Optional zero-based index to send a single record.",
    )
    parser.add_argument(
        "--agent-id",
        default="main",
        help="OpenClaw agent ID to run the prompt against (default: main).",
    )
    parser.add_argument(
        "--tool-helper-file",
        default=str(DEFAULT_TOOL_HELP_FILE),
        help="Optional helper file containing available local tool commands.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prompt_path = Path(args.prompt_file)
    tool_helper_path = Path(args.tool_helper_file)
    send_records_to_openclaw(
        prompt_path,
        record_index=args.record_index,
        agent_id=args.agent_id,
        tool_helper_file=tool_helper_path,
    )


if __name__ == "__main__":
    main()

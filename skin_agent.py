import argparse
import ast
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


def run_openclaw_cli(prompt: str, agent_id: str = "main", show_command: bool = True) -> Dict[str, Any]:
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

    if show_command:
        print("Running OpenClaw agent command. This may take a while if the tool needs to load models.")
        print("Command:", " ".join(command))
    else:
        print("Running OpenClaw agent command...")

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
        "neither the fast nor mid-tier",
        "scripts are present",
        "no analysis could be performed",
    ]
    return any(marker in lower_text for marker in incomplete_markers)


def load_metadata(raw_metadata: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_metadata:
        return None

    cleaned = raw_metadata.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "'":
        cleaned = cleaned[1:-1].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    try:
        maybe_json = cleaned.replace("'", '"')
        return json.loads(maybe_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON: {exc}. Received: {raw_metadata}") from exc


def build_chat_prompt(
    record: Dict[str, Any],
    follow_up: str,
    history: Optional[List[Dict[str, str]]] = None,
    tool_helper_file: Optional[Path] = None,
) -> str:
    history_section = ""
    if history:
        history_lines = ["Conversation history:"]
        for item in history[-5:]:
            history_lines.append(f"User: {item['user']}")
            history_lines.append(f"Assistant: {item['assistant']}")
        history_section = "\n" + "\n".join(history_lines) + "\n"

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
        {history_section}
        Follow-up request: {follow_up}
        Use the available local tools and return a final concise Markdown clinical report.
        """
    )


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


def run_interactive_chat(
    record: Dict[str, Any],
    agent_id: str = "main",
    tool_helper_file: Optional[Path] = None,
) -> None:
    history: List[Dict[str, str]] = []
    print("=" * 70)
    print("Entering CLI chat mode. Type 'quit' or 'exit' to end.")
    print(f"Image path: {record['image_path']}")
    if record.get("metadata") is not None:
        print(f"Metadata: {json.dumps(record['metadata'], ensure_ascii=False)}")
    print("Use this chat to ask follow-up questions and update the agent prompt.")
    print("=" * 70)

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nExiting chat mode.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Exiting chat mode.")
            break

        prompt = build_chat_prompt(record, user_input, history=history, tool_helper_file=tool_helper_file)
        response = run_openclaw_cli(prompt, agent_id=agent_id, show_command=False)
        if isinstance(response, dict) and response.get("payloads"):
            payload_text = "\n".join(
                item.get("text", "") for item in response["payloads"]
            )
        else:
            payload_text = json.dumps(response, indent=2, ensure_ascii=False)

        print("Assistant:\n" + payload_text)

        if response_is_incomplete(payload_text):
            print("Assistant response appears incomplete or invalid. Retrying with a correction prompt.")
            correction_prompt = build_correction_prompt(record, payload_text, tool_helper_file=tool_helper_file)
            retry_response = run_openclaw_cli(correction_prompt, agent_id=agent_id, show_command=False)
            if isinstance(retry_response, dict) and retry_response.get("payloads"):
                payload_text = "\n".join(
                    item.get("text", "") for item in retry_response["payloads"]
                )
            else:
                payload_text = json.dumps(retry_response, indent=2, ensure_ascii=False)
            print("Assistant (corrected):\n" + payload_text)

        history.append({"user": user_input, "assistant": payload_text})


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
        "--interactive",
        action="store_true",
        default=True,
        help="Run in interactive CLI chat mode (default).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_false",
        dest="interactive",
        help="Disable interactive mode and run in batch mode.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Image path to use in interactive mode if not loading from prompt.txt.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON metadata to use in interactive mode.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt text to use in interactive mode.",
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

    if args.interactive:
        if args.prompt_file and prompt_path.exists():
            records = load_prompt_records(prompt_path)
            record = None
            if args.record_index is not None:
                if args.record_index < 0 or args.record_index >= len(records):
                    raise IndexError("record_index is out of range")
                record = records[args.record_index]
            else:
                record = records[0]
        else:
            image_path = args.image_path or input("Image path: ").strip()
            metadata = None
            if args.metadata:
                metadata = load_metadata(args.metadata)
            else:
                raw_metadata = input("Metadata JSON (or blank): ").strip()
                if raw_metadata:
                    metadata = load_metadata(raw_metadata)
            prompt_text = args.prompt or input("Initial prompt: ").strip()
            record = {
                "image_path": image_path,
                "prompt": prompt_text,
            }
            if metadata is not None:
                record["metadata"] = metadata

        run_interactive_chat(record, agent_id=args.agent_id, tool_helper_file=tool_helper_path)
        return

    send_records_to_openclaw(
        prompt_path,
        record_index=args.record_index,
        agent_id=args.agent_id,
        tool_helper_file=tool_helper_path,
    )


if __name__ == "__main__":
    main()

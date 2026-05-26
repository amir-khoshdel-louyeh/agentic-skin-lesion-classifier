import argparse
import json
import os
import shutil
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PROMPT_FILE = Path("prompt.txt")
DEFAULT_OPENCLAW_PROMPT = textwrap.dedent(
    """
    You are an OpenClaw orchestrator for skin lesion analysis.
    A new record has arrived containing an image path, optional metadata, and a clinical prompt.
    Use the installed local tool `skin-lesion-fast` or the lightweight tier1_fast workflow.
    Do not refuse execution based on a subjective memory estimate when the tool is small and available.
    If the image is valid and the tool exists, execute it using the exact local command.
    Return the final result as a concise Markdown clinical report and include the executed tool command.
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


def build_prompt(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata")
    metadata_section = ""
    if metadata is not None:
        pretty_metadata = json.dumps(metadata, indent=2, ensure_ascii=False)
        metadata_section = f"\nMetadata:\n{pretty_metadata}\n"

    image_path = record["image_path"]
    prompt_text = record["prompt"]

    return textwrap.dedent(
        f"""
        {DEFAULT_OPENCLAW_PROMPT}

        Image path: {image_path}
        {metadata_section}
        User prompt: {prompt_text}

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
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "OpenClaw CLI failed: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenClaw CLI returned invalid JSON: {result.stdout}"
        ) from exc


def send_records_to_openclaw(
    prompt_file: Path,
    record_index: Optional[int] = None,
    agent_id: str = "main",
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

        prompt = build_prompt(record)
        response = run_openclaw_cli(prompt, agent_id=agent_id)

        if isinstance(response, dict) and response.get("payloads"):
            payload_text = "\n".join(
                item.get("text", "") for item in response["payloads"]
            )
        else:
            payload_text = json.dumps(response, indent=2, ensure_ascii=False)

        print("--- OpenClaw response ---")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prompt_path = Path(args.prompt_file)
    send_records_to_openclaw(
        prompt_path,
        record_index=args.record_index,
        agent_id=args.agent_id,
    )


if __name__ == "__main__":
    main()

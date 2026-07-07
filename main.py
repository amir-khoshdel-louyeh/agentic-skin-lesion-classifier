"""Entry point for the Agentic Skin Lesion Classifier (see proposal.tmp).

Thin CLI dispatcher over the orchestrator logic in skin_agent.py:
  python main.py --record-index 0    # single round
  python main.py --no-interactive    # all records, sequential
  python main.py --chat --record 3   # physician debate on record 3
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from skin_agent import (
    load_prompt_records,
    run_interactive_chat,
    send_records_to_openclaw,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic Skin Lesion Classifier — orchestrator entry point."
    )
    parser.add_argument(
        "--prompt-file",
        default=str(ROOT_DIR / "prompt.yaml"),
        help="Path to the prompt records file (relative or absolute).",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=None,
        help="Zero-based record index for single-round mode.",
    )
    parser.add_argument(
        "--record",
        type=int,
        default=None,
        help="Zero-based record index for chat mode.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Batch mode: process records sequentially without chat.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Physician debate mode on the selected record.",
    )
    parser.add_argument(
        "--agent-id",
        default="main",
        help="OpenClaw agent ID (default: main).",
    )
    parser.add_argument(
        "--tool-helper-file",
        default=str(ROOT_DIR / "tool_manifest.md"),
        help="Tool manifest path (relative or absolute).",
    )
    return parser


def resolve(path_str: str) -> Path:
    """Cross-platform path standard: normalize everything to absolute Paths."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def main() -> None:
    args = build_parser().parse_args()
    prompt_path = resolve(args.prompt_file)
    tool_helper_path = resolve(args.tool_helper_file)

    if args.chat:
        if args.record is None:
            raise SystemExit("--chat requires --record <index>.")
        records = load_prompt_records(prompt_path)
        if args.record < 0 or args.record >= len(records):
            raise SystemExit("record index out of range.")
        run_interactive_chat(
            records[args.record],
            agent_id=args.agent_id,
            tool_helper_file=tool_helper_path,
        )
        return

    if args.no_interactive or args.record_index is not None:
        send_records_to_openclaw(
            prompt_path,
            record_index=args.record_index,
            agent_id=args.agent_id,
            tool_helper_file=tool_helper_path,
        )
        return

    # Default: interactive chat on the first record (matches skin_agent.py).
    records = load_prompt_records(prompt_path)
    run_interactive_chat(
        records[0],
        agent_id=args.agent_id,
        tool_helper_file=tool_helper_path,
    )


if __name__ == "__main__":
    main()

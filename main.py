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
        "--doctor-check",
        action="store_true",
        help="Validate prompts/*.doctor.md (bounds 0.6-0.9, "
        "only-more-cautious, no safety overrides) and print the "
        "effective thresholds. Exits non-zero on any violation.",
    )
    parser.add_argument(
        "--agent-id",
        default="main",
        help="OpenClaw agent ID (default: main).",
    )
    parser.add_argument(
        "--tool-helper-file",
        default=str(ROOT_DIR / "tools" / "manifest.yaml"),
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
    if args.doctor_check:
        from control.prompts import doctor_check

        report = doctor_check()
        failed = False
        for role, info in report.items():
            status = "OK" if info["ok"] else "FAILED"
            print(f"[{role}] {status}")
            for name, value in info["effective"].items():
                print(f"  {name}: {value}")
            for err in info["errors"]:
                print(f"  ! {err}")
            failed = failed or not info["ok"]
        raise SystemExit(1 if failed else 0)
    prompt_path = resolve(args.prompt_file)
    tool_helper_path = resolve(args.tool_helper_file)

    if args.chat:
        if args.record is None:
            raise SystemExit("--chat requires --record <index>.")
        from control.chat import build_debate_prompt

        try:
            print(build_debate_prompt(args.record))
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc}\nHint: run the round first "
                f"(--record-index {args.record})."
            ) from exc
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
        from orchestrator import run_round

        if args.no_interactive and args.record_index is None:
            from control.decide import append_audit

            records = load_prompt_records(prompt_path)
            for index in range(len(records)):
                try:
                    out = run_round(
                        index,
                        prompt_file=str(prompt_path),
                        tool_helper_file=str(tool_helper_path),
                        agent_id=args.agent_id,
                    )
                    print(f"Round {index} report: {out}")
                except Exception as exc:
                    # One bad record must not kill the sweep.
                    append_audit({
                        "event": "round_failed",
                        "record_index": index,
                        "error": str(exc)[:500],
                    })
                    print(f"Round {index} FAILED ({exc}); continuing.")
                    from orchestrator import ping

                    ping(f"Round {index} FAILED: {str(exc)[:200]}",
                         level="error")
            return
        out = run_round(
            args.record_index or 0,
            prompt_file=str(prompt_path),
            tool_helper_file=str(tool_helper_path),
            agent_id=args.agent_id,
        )
        print(f"Round report: {out}")
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

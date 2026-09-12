"""Notification hook: desktop toast on round completion/failure.

Best-effort by design: tries `notify-send` (Linux), falls back to a
terminal bell + stderr line when headless. NEVER fails the caller —
exit 0 with `delivered: true/false` in the uniform JSON receipt, so a
missing desktop cannot break a round.
"""

import argparse
import json
import shutil
import subprocess
import sys

LEVELS = ("info", "warning", "error")
URGENCY = {"info": "normal", "warning": "normal", "error": "critical"}
MAX_MESSAGE = 500


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def send(title: str, message: str, level: str) -> dict:
    message = message if len(message) <= MAX_MESSAGE else message[:MAX_MESSAGE] + "…"
    backend = shutil.which("notify-send")
    if backend:
        try:
            proc = subprocess.run(
                [backend, "--urgency", URGENCY[level], "--app-name",
                 "skin-lesion", title, message],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                return {"delivered": True, "backend": "notify-send",
                        "fallback": False}
        except Exception:  # noqa: BLE001 - fall through to fallback
            pass
    # Headless fallback: visible on the operator console, never an error.
    print(f"[notify:{level}] {title}: {message}", file=sys.stderr)
    print("\a", end="", file=sys.stderr)
    return {"delivered": False, "backend": "console-fallback",
            "fallback": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Desktop toast hook.")
    parser.add_argument("--message", required=True)
    parser.add_argument("--level", default="info", choices=LEVELS)
    parser.add_argument("--title", default="skin-lesion round")
    args = parser.parse_args()
    if not args.message.strip():
        return fail("Empty --message.")
    result = send(args.title, args.message.strip(), args.level)
    print(json.dumps({
        "status": "success",
        "tool": "notify",
        "model_tier": "notification",
        "model_executed": "desktop_toast_hook",
        "level": args.level,
        "title": args.title,
        "message": args.message.strip()[:MAX_MESSAGE],
        "uncertainty_flags": [],
        **result,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

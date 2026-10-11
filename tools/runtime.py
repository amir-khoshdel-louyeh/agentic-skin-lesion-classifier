"""Re-exec a tool under the project venv when the caller's python lacks deps.

The OpenClaw agent runs tool lines as `python tools/<tool>.py ...` with
whatever interpreter its PATH resolves. That interpreter is not always
the project venv (the round report showed a torchvision ImportError from
`/home/amir/.openclaw/workspace`), so most of the pool failed before
inference. Every tool calls `ensure_runtime()` before importing heavy
libraries: if a dependency is missing, the same argv is re-executed with
`.venv/bin/python` once. Control-side calls already use sys.executable,
so this is a no-op there.
"""

import importlib.util
import os
import sys
from pathlib import Path

BOOTSTRAP_ENV = "SKIN_RUNTIME_BOOTSTRAPPED"


def _missing(deps: tuple[str, ...]) -> list[str]:
    return [name for name in deps
            if importlib.util.find_spec(name) is None]


def ensure_runtime(deps: tuple[str, ...] = ("torch", "torchvision",
                                            "PIL", "yaml")) -> None:
    """Re-exec under .venv/bin/python if `deps` are missing (once)."""
    if os.environ.get(BOOTSTRAP_ENV):
        return
    if not _missing(deps):
        return
    root = Path(__file__).resolve().parent.parent
    candidates = [root / ".venv" / "bin" / "python",
                  root / ".venv" / "Scripts" / "python.exe"]
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if not candidate.exists() or candidate.resolve() == current:
            continue
        os.environ[BOOTSTRAP_ENV] = "1"
        os.execv(str(candidate), [str(candidate), *sys.argv])
    # No usable venv: fall through and let the tool report the real error.

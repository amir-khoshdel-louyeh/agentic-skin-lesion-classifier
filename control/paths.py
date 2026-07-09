"""Cross-platform path standard (proposal.tmp Section 11).

All path-bearing files use repo-relative paths. Absolute literals
(e.g. C:/..., /home/...) fail startup with a clear error instead of
failing mid-round on another OS.
"""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

_ABSOLUTE_RE = re.compile(
    r"(?<![A-Za-z0-9+_.-])[A-Za-z]:[\\/]"
    r"|(?<![\w/.-])\\\\"
    r"|(?<![\w/.:-])/(?:home|Users|tmp|etc|mnt|data|root)\b"
)

SCANNED_SUFFIXES = (".yaml", ".yml", ".md")


def resolve(path_str: str, root: Path = ROOT_DIR) -> Path:
    """Normalize a user- or file-supplied path to an absolute Path."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def find_absolute_literals(text: str) -> list[str]:
    return sorted(set(_ABSOLUTE_RE.findall(text)))


def startup_path_check(root: Path = ROOT_DIR) -> None:
    """Fail fast if any tracked config/doc file holds absolute literals."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        try:
            hits = find_absolute_literals(
                path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue
        if hits:
            offenders[str(path.relative_to(root))] = hits
    if offenders:
        details = "; ".join(
            f"{name}: {', '.join(hits)}" for name, hits in offenders.items()
        )
        raise SystemExit(f"Absolute path literals forbidden: {details}")

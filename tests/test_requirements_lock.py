"""Dependency lock contract tests."""

import re
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_lock_is_complete_and_fully_pinned():
    lock_path = ROOT / "requirements.lock"
    assert lock_path.is_file()
    lines = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines
    assert all(re.fullmatch(r"[A-Za-z0-9_.+-]+==[^\s=]+", line)
               for line in lines), lines

    locked = {canonical(line.split("==", 1)[0]) for line in lines}
    required = set()
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            required.add(canonical(Requirement(line).name))

    assert required <= locked, sorted(required - locked)
    assert "pytest" in locked
    assert "torch" in locked
    assert "python-multipart" in locked
    assert not any("@" in line for line in lines)

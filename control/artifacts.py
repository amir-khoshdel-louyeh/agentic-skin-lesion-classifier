"""Model and calibration artefact integrity manifest.

The manifest is intentionally content-addressed and timestamp-free: the
same files produce the same JSON. Calibration files are required because
they determine which tools enter the diagnostic pool. Model files are
hashed when present but treated as optional by default, since weights are
git-ignored and may be absent on a clean checkout.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from control.paths import ROOT_DIR

SCHEMA_VERSION = 1
DEFAULT_MANIFEST = "artifacts/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _model_files(root: Path) -> list[Path]:
    directory = root / "models"
    if not directory.exists():
        return []
    files = []
    for path in sorted(directory.rglob("*")):
        if (not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"):
            continue
        files.append(path)
    return files


def build_manifest(root: Path = ROOT_DIR) -> dict:
    root = Path(root)
    calibration = {
        entry["path"]: {"sha256": entry["sha256"], "bytes": entry["bytes"]}
        for entry in (
            _entry(path, root)
            for path in sorted((root / "calibration").glob("*"))
            if path.is_file()
        )
    }
    models = {
        entry["path"]: {"sha256": entry["sha256"], "bytes": entry["bytes"]}
        for entry in (_entry(path, root) for path in _model_files(root))
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "calibration": calibration,
        "models": models,
    }


def write_manifest(path: Path, root: Path = ROOT_DIR) -> dict:
    manifest = build_manifest(root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return manifest


def _safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest path: {relative}")
    return root / path


def verify_manifest(
    root: Path = ROOT_DIR,
    manifest_path: Path | None = None,
    *,
    allow_missing_models: bool = True,
) -> dict:
    """Check every listed artefact and report tampering or omissions."""
    root = Path(root)
    manifest_path = Path(manifest_path or (root / DEFAULT_MANIFEST))
    problems: list[str] = []
    warnings: list[str] = []
    checked = 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "checked": 0,
            "problems": [f"unreadable manifest: {exc}"],
            "warnings": [],
        }

    calibration = manifest.get("calibration") or {}
    models = manifest.get("models") or {}
    for relative, expected in sorted(calibration.items()):
        try:
            path = _safe_path(root, relative)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if not path.is_file():
            problems.append(f"missing calibration artifact: {relative}")
            continue
        if path.stat().st_size != expected.get("bytes"):
            problems.append(f"size mismatch: {relative}")
        if sha256_file(path) != expected.get("sha256"):
            problems.append(f"sha256 mismatch: {relative}")
        checked += 1

    for relative, expected in sorted(models.items()):
        try:
            path = _safe_path(root, relative)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if not path.is_file():
            message = f"missing model artifact: {relative}"
            (warnings if allow_missing_models else problems).append(message)
            continue
        if path.stat().st_size != expected.get("bytes"):
            problems.append(f"size mismatch: {relative}")
        if sha256_file(path) != expected.get("sha256"):
            problems.append(f"sha256 mismatch: {relative}")
        checked += 1

    present_calibration = {
        path.relative_to(root).as_posix()
        for path in sorted((root / "calibration").glob("*"))
        if path.is_file()
    }
    for relative in sorted(present_calibration - set(calibration)):
        problems.append(f"unlisted calibration artifact: {relative}")

    present_models = {
        path.relative_to(root).as_posix() for path in _model_files(root)
    }
    for relative in sorted(present_models - set(models)):
        warnings.append(f"unlisted model artifact: {relative}")

    return {
        "ok": not problems,
        "manifest": str(manifest_path),
        "checked": checked,
        "problems": problems,
        "warnings": warnings,
    }

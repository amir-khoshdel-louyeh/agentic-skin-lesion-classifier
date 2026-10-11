"""Stage case images where the OpenClaw agent runtime is allowed to read them.

Why this exists
---------------
OpenClaw refuses to read a local media path unless its *realpath* sits under
one of its allowed roots (its ``dist/local-media-access-*.js`` raises
``Local media path is not under an allowed directory``). Those roots are the
OpenClaw tmp dir, ``<config>/media``, ``<state>/{media,canvas,workspace,
sandboxes}`` and the agent workspace. A *symlink* from the agent workspace
into this repo resolves outside every root, so the agent receives an error
instead of the image, and the Quality Agent can never run its toolkit.

The remedy is a real directory inside the agent workspace whose files share
inodes with the repo dataset: same bytes, no extra disk space on one
filesystem, and in-place edits are picked up by both paths. Directory
*entries* are a snapshot, though — a file added to the repo afterwards does
not appear until this script runs again. Hence this helper: run it whenever
the dataset changes (manually, or from cron).

Usage
-----
  python stage_openclaw_media.py                 # add/update the default target
  python stage_openclaw_media.py --dry-run       # list what would change
  python stage_openclaw_media.py --check         # exit 1 if out of date
  python stage_openclaw_media.py --prune         # also drop stale target entries
  python stage_openclaw_media.py --target DIR    # custom staging directory

`--check` compares both directions (missing/updated *and* extra entries) but
never writes, so it is safe for cron/CI: exit 0 means the staging directory
is a true mirror of the dataset.

The default target is ``$OPENCLAW_STATE_DIR/workspace/dataset`` (falling back
to ``~/.openclaw/workspace/dataset``). Nothing outside the target directory is
ever modified, and ``--prune`` is required to delete anything.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT_DIR / "dataset"


def default_target() -> Path:
    state = os.environ.get("OPENCLAW_STATE_DIR") or "~/.openclaw"
    return Path(state).expanduser() / "workspace" / "dataset"


def _same_file(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev)


def _contains(parent: Path, child: Path) -> bool:
    """True when `child` is `parent` or lives underneath it."""
    parent, child = parent.resolve(), child.resolve()
    return child == parent or parent in child.parents


def _guard_paths(source: Path, target: Path) -> None:
    """Refuse layouts that could delete real data.

    Both containment directions are refused: target inside source would
    re-process its own output, and source inside target would let `--prune`
    walk the source tree and delete it.
    """
    if source.is_symlink():
        raise SystemExit(f"Source {source} is a symlink; refusing to run.")
    if target.is_symlink():
        raise SystemExit(
            f"Target {target} is a symlink — that is the very thing this "
            "helper replaces. Remove it first (rm <target>) and re-run."
        )
    if target.exists() and not target.is_dir():
        raise SystemExit(f"Target {target} exists and is not a directory.")
    if target.resolve() == source.resolve():
        raise SystemExit("Target is the source dataset; refusing to run.")
    if _contains(source, target):
        raise SystemExit(f"Target {target} is inside the source; refusing.")
    if _contains(target, source):
        raise SystemExit(
            f"Target {target} contains the source dataset — `--prune` could "
            "delete real data there; refusing to run."
        )


def sync(source: Path = DEFAULT_SOURCE, target: Path | None = None,
         prune: bool = False, dry_run: bool = False) -> dict:
    """Add/update `target` from `source`. Returns a summary dict.

    Linking is preferred (shared inode, zero extra bytes); a cross-device or
    filesystem-refused link falls back to a byte copy, which is reported.
    Never raises on individual files: problems accumulate in `errors`.
    """
    target = Path(target) if target is not None else default_target()
    _guard_paths(source, target)
    if not source.is_dir():
        raise SystemExit(f"Source dataset not found: {source}")

    added, updated, replaced, copied, pruned = [], [], [], [], []
    errors: list[str] = []

    for src in sorted(source.rglob("*")):
        rel = src.relative_to(source)
        dst = target / rel
        if src.is_dir():
            if not dst.is_dir() and not dry_run:
                dst.mkdir(parents=True, exist_ok=True)
            continue
        if not src.is_file():
            continue
        if dst.exists() and _same_file(src, dst):
            continue
        existed = dst.exists()
        if dry_run:
            (updated if existed else added).append(str(rel))
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if existed:
                dst.unlink()
            try:
                os.link(src, dst)  # hardlink: same bytes, no extra space
                (replaced if existed else added).append(str(rel))
            except OSError:
                shutil.copy2(src, dst)  # cross-device fallback
                copied.append(str(rel))
        except OSError as exc:
            errors.append(f"{rel}: {exc}")

    if prune and target.is_dir():
        for dst in sorted(target.rglob("*"), reverse=True):
            rel = dst.relative_to(target)
            src = source / rel
            try:
                if dst.is_dir() and not dst.is_symlink():
                    if not src.is_dir() and not any(dst.iterdir()):
                        if dry_run:
                            pruned.append(f"{rel}/")
                        else:
                            dst.rmdir()
                    continue
                if dst.exists() and not src.exists():
                    if dry_run:
                        pruned.append(str(rel))
                    else:
                        dst.unlink()
                        pruned.append(str(rel))
            except OSError as exc:
                errors.append(f"{rel}: {exc}")

    return {
        "source": str(source),
        "target": str(target),
        "dry_run": dry_run,
        "added": added,
        "updated": replaced,
        "copied_fallback": copied,
        "pruned": pruned,
        "errors": errors,
        "in_sync": not (added or replaced or copied or pruned or errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage repo case images into an OpenClaw-readable "
        "directory (hardlinks; see module docstring for why).")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Dataset directory to stage (default: ./dataset).")
    parser.add_argument("--target", default=None,
                        help="Staging directory (default: "
                        "$OPENCLAW_STATE_DIR/workspace/dataset).")
    parser.add_argument("--prune", action="store_true",
                        help="Also delete target entries that no longer exist "
                        "in the source (off by default).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing.")
    parser.add_argument("--check", action="store_true",
                        help="Write nothing; exit 1 when the target is not a "
                        "true mirror (missing/updated or extra entries).")
    parser.add_argument("--quiet", action="store_true",
                        help="Print the JSON summary only.")
    args = parser.parse_args()

    summary = sync(
        source=Path(args.source).expanduser(),
        target=(Path(args.target).expanduser() if args.target
                else default_target()),
        # `--check` implies the prune *scan* so "up to date" means a true
        # mirror in both directions: an entry that exists only in the target
        # is drift too, and would otherwise be reported as up to date. The
        # scan never deletes anything here, because --check sets dry_run.
        prune=args.prune or args.check,
        dry_run=args.dry_run or args.check,
    )

    if not args.quiet:
        verb = "would change" if summary["dry_run"] else "changed"
        print(f"source: {summary['source']}")
        print(f"target: {summary['target']}")
        print(f"{verb}: added={len(summary['added'])} "
              f"updated={len(summary['updated'])} "
              f"copied={len(summary['copied_fallback'])} "
              f"pruned={len(summary['pruned'])}")
        for key in ("added", "updated", "copied_fallback", "pruned", "errors"):
            for item in summary[key][:20]:
                print(f"  {key}: {item}")
            if len(summary[key]) > 20:
                print(f"  {key}: ... {len(summary[key]) - 20} more")
        if summary["in_sync"]:
            print("staging is up to date.")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("added", "updated", "copied_fallback",
                                   "pruned")},
                     indent=2))

    if args.check:
        return 1 if not summary["in_sync"] else 0
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

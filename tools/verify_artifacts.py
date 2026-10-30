"""Write or verify the model/calibration integrity manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.artifacts import (  # noqa: E402
    DEFAULT_MANIFEST,
    verify_manifest,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write or verify model/calibration SHA-256 manifest."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true",
                        help="Regenerate the manifest from current files.")
    parser.add_argument("--strict-models", action="store_true",
                        help="Fail when a listed model file is missing.")
    args = parser.parse_args()

    path = ROOT_DIR / args.manifest
    if args.write:
        manifest = write_manifest(path, ROOT_DIR)
        print(json.dumps({
            "ok": True,
            "manifest": str(path),
            "calibration": len(manifest["calibration"]),
            "models": len(manifest["models"]),
        }, indent=2))
        return 0

    result = verify_manifest(
        ROOT_DIR, path, allow_missing_models=not args.strict_models
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

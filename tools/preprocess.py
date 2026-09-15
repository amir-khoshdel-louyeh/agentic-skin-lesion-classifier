"""Preprocessing slot: hair removal + denoise, image-in -> image-out.

Activates the extension slot reserved in proposal.tmp Section 5:
category `preprocessing`, discovered at runtime via Manifest v2
(`capable_of("cleaned_image", ["image"])`) with zero orchestrator/agent
code changes. CPU-only OpenCV, deterministic, no weights.

Chain rule (role assignment, report/phase6_benchmark_preprocess.md):
quality-gate flags `blurry` -> run this tool -> re-run quality-gate ->
continue the round with the cleaned image. Agents chain it; the
orchestrator receipt check covers `python tools/*.py` commands.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.quality_gate import assess as quality_assess  # noqa: E402


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def remove_hair(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """DullRazor-style: black-hat + keep only long thin components.

    Threshold 10 fires on lesion texture (3-10% of pixels on clean
    frames); 25 + elongation filter keeps only hair-like structures so
    clean images pass through nearly untouched (do-no-harm gate).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, raw = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    hair_mask = np.zeros_like(raw)
    for i in range(1, ncomp):
        wpx, hpx = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        long_side, short_side = max(wpx, hpx), max(min(wpx, hpx), 1)
        if long_side >= 30 and long_side / short_side >= 3.0:
            hair_mask[labels == i] = 255
    n_hair_px = int((hair_mask > 0).sum())
    if n_hair_px == 0:
        return bgr, 0
    cleaned = cv2.inpaint(bgr, hair_mask, 1, cv2.INPAINT_TELEA)
    return cleaned, n_hair_px


def denoise(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hair removal + denoise preprocessing (image-in, image-out)."
    )
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument(
        "--steps",
        default="hair,denoise",
        help="Comma subset of {hair,denoise} (default: hair,denoise).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    wanted = [s.strip() for s in args.steps.split(",") if s.strip()]
    if not wanted or any(s not in ("hair", "denoise") for s in wanted):
        return fail("Invalid --steps (use a subset of hair,denoise).")

    image = cv2.imread(args.image_path)
    if image is None:
        return fail(f"Unreadable: {args.image_path}")
    before = quality_assess(args.image_path)

    out = image
    hair_px = 0
    applied: list[str] = []
    if "hair" in wanted:
        out, hair_px = remove_hair(out)
        applied.append("hair_removal")
    if "denoise" in wanted:
        out = denoise(out)
        applied.append("denoise")

    out_path = Path(args.output_path).expanduser()
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out):
        return fail(f"Could not write: {out_path}")
    after = quality_assess(str(out_path))

    print(json.dumps({
        "status": "success",
        "tool": "preprocess",
        "model_tier": "preprocessing",
        "model_executed": "opencv_hair_denoise",
        "input": args.image_path,
        "output": str(out_path.relative_to(BASE_DIR))
        if out_path.is_relative_to(BASE_DIR) else str(out_path),
        "steps_applied": applied,
        "hair_pixels_inpainted": hair_px,
        "quality_before": before.get("uncertainty_flags"),
        "quality_after": after.get("uncertainty_flags"),
        "quality_pass_before": before.get("quality_pass"),
        "quality_pass_after": after.get("quality_pass"),
        "uncertainty_flags": [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

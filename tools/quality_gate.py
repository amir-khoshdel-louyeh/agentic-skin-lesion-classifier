"""Image quality gate (utility tier, no weights).

OpenCV checks: blur (Laplacian variance), exposure (brightness +
clipping), resolution. Flags unusable images BEFORE diagnosis so agents
can report "cannot assess" honestly instead of guessing.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

MIN_SIDE_PX = 128
BLUR_THRESHOLD = 60.0
BRIGHT_LO = 30.0
BRIGHT_HI = 225.0
CLIP_FRACTION = 0.25


def assess(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": f"Unreadable: {image_path}"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(gray.mean())
    dark_frac = float((gray < 8).mean())
    bright_frac = float((gray > 247).mean())
    flags: list[str] = []
    if min(h, w) < MIN_SIDE_PX:
        flags.append("low_resolution")
    if blur < BLUR_THRESHOLD:
        flags.append("blurry")
    if not BRIGHT_LO <= mean_bright <= BRIGHT_HI:
        flags.append("bad_exposure")
    if dark_frac + bright_frac > CLIP_FRACTION:
        flags.append("clipped")
    return {
        "status": "success",
        "tool": "quality-gate",
        "model_tier": "utility",
        "model_executed": "opencv_quality_gate",
        "quality_pass": not flags,
        "metrics": {
            "width": w,
            "height": h,
            "blur_variance": round(blur, 2),
            "mean_brightness": round(mean_bright, 2),
            "clipped_fraction": round(dark_frac + bright_frac, 4),
        },
        "uncertainty_flags": flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Image quality gate.")
    parser.add_argument("--image", dest="image_path", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error",
                           "message": f"Image not found: {args.image_path}"}))
        return 1
    print(json.dumps(assess(args.image_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ABCDE heuristic analyzer (Phase 6, non-ML support tier).

Deterministic OpenCV-only lesion heuristic: asymmetry, border
irregularity, color variegation, diameter + composite score. No weights,
no learning, CPU-only (0 GB VRAM). Uniform JSON contract so agents can
call it exactly like the diagnostic tools.

Output is a morphological flag, NOT a diagnosis: high scores mean
"dermoscopy features worth a second tier", never a malignancy verdict.
"""

import argparse
import json
import os

import cv2
import numpy as np


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def segment_lesion(bgr: np.ndarray) -> np.ndarray | None:
    """Largest central dark blob via Otsu; None when nothing plausible."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Clean small specks; keep the dominant blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape
    biggest = max(contours, key=cv2.contourArea)
    area_frac = float(cv2.contourArea(biggest) / (h * w))
    if area_frac < 0.005 or area_frac > 0.95:
        # Otsu picked the background (whole-frame mask): invert once and
        # retry. Common on evenly lit dermoscopy frames. Deterministic.
        inv = cv2.bitwise_not(cleaned)
        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        area_frac = float(cv2.contourArea(biggest) / (h * w))
        if area_frac < 0.005 or area_frac > 0.95:
            return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [biggest], -1, 255, cv2.FILLED)
    return mask


def asymmetry_score(mask: np.ndarray) -> float:
    """Mean non-overlap after horizontal + vertical flip (0=symmetric)."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    hflip = cv2.flip(m, 1)
    vflip = cv2.flip(m, 0)
    # Center the mask before comparing so off-center lesions are not
    # penalized for position rather than shape.
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return 0.0
    cy, cx = int(ys.mean()), int(xs.mean())
    canvas = np.zeros((h * 2, w * 2), dtype=np.uint8)
    oy, ox = h - cy, w - cx
    canvas[oy:oy + h, ox:ox + w] = m
    c = canvas[h - h // 2:h + h // 2, w - w // 2:w + w // 2]
    hf = cv2.flip(c, 1)
    vf = cv2.flip(c, 0)
    union_h = np.logical_or(c, hf).sum()
    union_v = np.logical_or(c, vf).sum()
    diff = (np.logical_xor(c, hf).sum() + np.logical_xor(c, vf).sum()) / 2.0
    union = (union_h + union_v) / 2.0
    return round(float(diff / union) if union else 0.0, 4)


def border_score(mask: np.ndarray) -> float:
    """Compactness irregularity: perimeter^2/(4*pi*area), 1=circle."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if area <= 0:
        return 0.0
    return round((perim * perim) / (4.0 * np.pi * area), 4)


def color_score(bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Variegation via HSV channel std + count of distinct color zones."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lesion = hsv[mask > 0]
    if len(lesion) == 0:
        return {"std_h": 0.0, "std_s": 0.0, "std_v": 0.0, "zones": 0, "score": 0.0}
    std_h, std_s, std_v = (round(float(lesion[:, i].std()), 2) for i in range(3))
    # Quantize hue into 6 zones + dark zone; count populated zones.
    zones = set()
    for h, s, v in lesion[:: max(1, len(lesion) // 4000)]:
        if v < 40:
            zones.add("dark")
        else:
            zones.add(int(h) // 30)
    n_zones = len(zones)
    # 0..1 heuristic: hue spread + saturation spread + zone count.
    zone_part = min(1.0, n_zones / 4.0)
    spread_part = min(1.0, (std_h / 40.0 + std_s / 60.0) / 2.0)
    combined = round((zone_part + spread_part) / 2.0, 4)
    return {
        "std_h": std_h,
        "std_s": std_s,
        "std_v": std_v,
        "zones": n_zones,
        "score": combined,
    }


def diameter_px(mask: np.ndarray) -> dict:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    major = 2.0 * float(radius)
    h, w = mask.shape
    return {
        "pixels": round(major, 1),
        "relative": round(float(major / max(h, w)), 4),
    }


def assess(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": f"Unreadable: {image_path}"}
    mask = segment_lesion(image)
    if mask is None:
        return {
            "status": "success",
            "tool": "abcde-analyzer",
            "model_tier": "support",
            "model_executed": "opencv_abcde_heuristic",
            "segmented": False,
            "abcde": None,
            "abcde_score": None,
            "risk_band": "unknown",
            "uncertainty_flags": ["no_segmentation"],
        }
    asym = asymmetry_score(mask)
    border = border_score(mask)
    color = color_score(image, mask)
    diam = diameter_px(image.shape[:2] and mask)
    # Sub-scores scaled to ~0..2 each (total 0..8). Cutoffs calibrated
    # on the 24 local ISIC2019 images so bands discriminate (see
    # report/phase6_benchmark_abcde.md); dermoscopy close-ups saturate
    # naive ABCDE mappings to "high" for every frame.
    a_pts = min(2.0, asym * 3.0)
    b_pts = min(2.0, max(0.0, (border - 1.0)) * 0.4)
    c_pts = min(2.0, color["score"] * 2.0)
    d_pts = min(2.0, diam["relative"] * 2.0)
    total = round(a_pts + b_pts + c_pts + d_pts, 2)
    band = "low" if total < 3.0 else ("moderate" if total <= 5.0 else "high")
    return {
        "status": "success",
        "tool": "abcde-analyzer",
        "model_tier": "support",
        "model_executed": "opencv_abcde_heuristic",
        "segmented": True,
        "abcde": {
            "asymmetry": asym,
            "border_irregularity": border,
            "color": color,
            "diameter": diam,
            "subscores": {
                "A": round(a_pts, 2),
                "B": round(b_pts, 2),
                "C": round(c_pts, 2),
                "D": round(d_pts, 2),
            },
        },
        "abcde_score": total,
        "risk_band": band,
        "uncertainty_flags": [] if band == "low" else ["borderline"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ABCDE heuristic analyzer.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")
    result = assess(args.image_path)
    if metadata:
        result["metadata"] = metadata
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

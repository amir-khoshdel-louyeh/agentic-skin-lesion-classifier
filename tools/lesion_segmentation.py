"""Lesion boundary segmentation with MedSAM (ViT-B, SAM architecture).

MedSAM (wanglab/medsam-vit-base, Apache-2.0) is prompt-driven: a box
around the region of interest is the prompt here. Dermoscopy frames are
lesion-centred, so a generous central box plus the model's own IoU score
selects the best of the three candidate masks. Output is a mask PNG plus
boundary metrics (area, compactness, solidity, centroid offset, border
contact); this is a SUPPORT tool, not a classifier, so it is not in the
diagnostic vote pool.

Uniform JSON contract; VRAM released in finally with budget check.
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.runtime import ensure_runtime  # noqa: E402

ensure_runtime(("torch", "cv2", "PIL"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

WEIGHTS_DIR = BASE_DIR / "models" / "medsam-vit-base"
VRAM_BUDGET_BYTES = 3 * 1024**3


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def mask_metrics(mask: np.ndarray) -> dict:
    h, w = mask.shape
    area = int(mask.sum())
    frac = area / float(h * w)
    metrics = {
        "width": w,
        "height": h,
        "mask_area_px": area,
        "area_fraction": round(frac, 4),
        "touches_border": bool(mask[0, :].any() or mask[-1, :].any()
                               or mask[:, 0].any() or mask[:, -1].any()),
        "bbox": None,
        "compactness": None,
        "solidity": None,
        "border_index": None,
        "centroid_offset": None,
    }
    if area == 0:
        return metrics
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    metrics["bbox"] = {"x0": round(x0 / w, 4), "y0": round(y0 / h, 4),
                       "x1": round(x1 / w, 4), "y1": round(y1 / h, 4)}
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        if perimeter > 0:
            metrics["compactness"] = round(
                4 * np.pi * contour_area / perimeter ** 2, 4)
            metrics["border_index"] = round(
                perimeter / (2 * np.sqrt(np.pi * max(contour_area, 1.0))), 4)
        hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
        if hull_area > 0:
            metrics["solidity"] = round(contour_area / hull_area, 4)
    cy, cx = float(ys.mean()), float(xs.mean())
    metrics["centroid_offset"] = {
        "dx": round(cx / w - 0.5, 4), "dy": round(cy / h - 0.5, 4)}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="MedSAM lesion segmentation.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    parser.add_argument("--mask-dir", default="report/segmentation")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    if not (WEIGHTS_DIR / "config.json").exists():
        return fail(f"MedSAM weights missing at: {WEIGHTS_DIR}")
    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")

    from control.vram import check as vram_check  # noqa: E402

    ok, reason = vram_check(1.5)
    if not ok:
        return fail(f"VRAM guard: {reason}")

    from transformers import SamModel, SamProcessor  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = (torch.cuda.memory_allocated(device)
                   if device.type == "cuda" else 0)
    model = None
    try:
        processor = SamProcessor.from_pretrained(str(WEIGHTS_DIR),
                                                 local_files_only=True)
        model = SamModel.from_pretrained(str(WEIGHTS_DIR),
                                         local_files_only=True)
        model.to(device).eval()

        image = ImageOps.exif_transpose(
            Image.open(args.image_path)).convert("RGB")
        width, height = image.size
        box = [[int(0.08 * width), int(0.08 * height),
                int(0.92 * width), int(0.92 * height)]]
        inputs = processor(image, input_boxes=[box], return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]
        scores = outputs.iou_scores[0, 0].float().cpu().numpy()
        best = int(scores.argmax())
        mask = masks[best].numpy() > 0.0
        metrics = mask_metrics(mask)
        flags: list[str] = []
        if not mask.any():
            flags.append("mask_empty")
        else:
            if metrics["area_fraction"] < 0.003:
                flags.append("lesion_too_small")
            if metrics["area_fraction"] > 0.85:
                flags.append("lesion_too_large")
            if metrics["touches_border"]:
                flags.append("touches_border")
            if (metrics["border_index"] or 0) > 2.5:
                flags.append("irregular_border")
        if float(scores[best]) < 0.6:
            flags.append("low_mask_confidence")

        mask_dir = BASE_DIR / args.mask_dir
        mask_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image_path).stem
        mask_path = mask_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))

        print(json.dumps({
            "status": "success",
            "tool": "lesion-segmentation",
            "model_tier": "support",
            "model_executed": "medsam_vit_base",
            "prompt_box": [round(v, 4) for v in
                           (0.08, 0.08, 0.92, 0.92)],
            "mask_iou_score": round(float(scores[best]), 4),
            "mask_path": str(mask_path.relative_to(BASE_DIR)),
            "metrics": metrics,
            "uncertainty_flags": flags,
            "metadata": metadata,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated(device) - vram_before > VRAM_BUDGET_BYTES:
                print(json.dumps({"status": "warning",
                                  "message": "VRAM budget exceeded."}),
                      file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

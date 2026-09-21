"""Mid-tier classifier: DermAI EfficientNet-B0 (models/dermai-b0).

Uses the vendored transformers-native EfficientNet-B0 (model.safetensors,
224px) — no download needed. Uniform JSON contract.
VRAM discipline: load -> infer -> release in finally.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "models" / "dermai-b0"

ISIC_TO_CLASS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
    "mel": "Melanoma",
}

VRAM_BUDGET_BYTES = 2 * 1024**3

# Calibrated by temperature scaling: T=4.9 fitted by NLL grid search on a
# 105-image ISIC2019_full split (15/class, SCC excluded, 2026-09-29).
# Accuracy unchanged (argmax invariant); mean confidence 0.91 -> 0.48,
# now tracking the ~0.62 empirical accuracy. NLL 2.80 -> 1.50.
TEMPERATURE = 4.9


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Mid-Tier EfficientNet-B0 classifier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error", "message": f"Image not found: {args.image_path}"}))
        return 1
    if not (WEIGHTS_DIR / "model.safetensors").exists():
        return fail(f"Mid-tier weights missing at: {WEIGHTS_DIR}")

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")

    try:
        import torch
        from transformers import AutoImageProcessor, EfficientNetForImageClassification
    except ImportError as exc:
        return fail(f"Missing dependency: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    model = None
    try:
        processor = AutoImageProcessor.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
        model = EfficientNetForImageClassification.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True).to(device).eval()
        image = ImageOps.exif_transpose(Image.open(args.image_path)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits / TEMPERATURE, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
        idx = int(class_idx.item())
        id2label = model.config.id2label
        code = id2label.get(idx, id2label.get(str(idx)))
        result = {
            "status": "success",
            "tool": "skin-lesion-mid",
            "model_tier": "tier2_mid",
            "model_executed": "dermai_efficientnet_b0_ham10000",
            "predicted_class_index": idx,
            "disease_name": ISIC_TO_CLASS[code],
            "confidence_score": round(float(confidence.item()), 4),
            "metadata": metadata,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated(device) - vram_before > VRAM_BUDGET_BYTES:
                print(json.dumps({"status": "warning", "message": "VRAM budget exceeded."}), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

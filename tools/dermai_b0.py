"""Uniform CLI wrapper: DermAI EfficientNet-B0, HAM10000 7-class (Phase 7).

Ready-made candidate (sgonzalez2000/dermai-efficientnet-b0, Apache-2.0):
transformers-native EfficientNet-B0, 224px, ISIC-code head. Same uniform
JSON contract as the other diagnostic tools. VRAM discipline: load ->
infer -> release in finally (del + empty_cache) with budget check.
Sequential GPU schedule: never concurrent with the Ollama LLM.
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


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="DermAI EfficientNet-B0 tier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    if not (WEIGHTS_DIR / "model.safetensors").exists():
        return fail(f"DermAI weights missing at: {WEIGHTS_DIR}")

    metadata: dict = {}
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
        processor = AutoImageProcessor.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True)
        model = EfficientNetForImageClassification.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True).to(device).eval()
        image = ImageOps.exif_transpose(
            Image.open(args.image_path)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
        idx = int(class_idx.item())
        id2label = model.config.id2label
        code = id2label.get(idx, id2label.get(str(idx)))
        result = {
            "status": "success",
            "tool": "dermai-b0",
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
                print(json.dumps({"status": "warning",
                                  "message": "VRAM budget exceeded."}),
                      file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

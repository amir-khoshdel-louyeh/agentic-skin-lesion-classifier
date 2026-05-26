import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core import ModelFactory, ImageProcessor, LABEL_MAP

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def validate_image_path(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {
            "status": "error",
            "message": f"Image not found at: {image_path}"
        }
    if not path.is_file():
        return {
            "status": "error",
            "message": f"Path exists but is not a file: {image_path}"
        }
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return {
            "status": "error",
            "message": f"Unsupported file type: {path.suffix}. Supported types: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        }

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        return {
            "status": "error",
            "message": f"File is not a valid image or is corrupted: {image_path}. {str(exc)}"
        }

    return {"status": "ok"}


def predict_fast(image_path: str) -> dict:
    validation = validate_image_path(image_path)
    if validation["status"] != "ok":
        return validation

    image_tensor = ImageProcessor.process_image(image_path, target_size=224)
    model = ModelFactory.get_model("efficientnet_b0", num_classes=7)
    probabilities = ModelFactory.run_inference(model, image_tensor)

    confidence, class_idx = probabilities.max(dim=1)
    idx = class_idx.item()

    return {
        "status": "success",
        "tool": "skin-lesion-fast",
        "model_tier": "tier1_fast",
        "model_executed": "efficientnet_b0",
        "predicted_class_index": idx,
        "disease_name": LABEL_MAP.get(idx, "Unknown Pathological Condition"),
        "confidence_score": round(confidence.item(), 4)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/skin_lesion_fast.py",
        description="Run the fast skin lesion screening model using EfficientNet-B0."
    )
    parser.add_argument("--image", dest="image_path", required=True, help="Path to the lesion image.")
    args = parser.parse_args()

    result = predict_fast(args.image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

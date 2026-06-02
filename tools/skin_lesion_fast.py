import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image, ImageOps
import torch
import timm
from torchvision import transforms

LABEL_MAP = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanocytic nevi",
    5: "Melanoma",
    6: "Vascular lesions",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


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


def parse_metadata(metadata_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not metadata_json:
        return None

    cleaned = metadata_json.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "'":
        cleaned = cleaned[1:-1].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    try:
        maybe_json = cleaned.replace("'", '"')
        return json.loads(maybe_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid metadata JSON: {exc}. Received: {metadata_json}"
        ) from exc


def process_image(image_path: str, target_size: int = 224):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_size = (int(width * scale), int(height * scale))
    image = image.resize(resized_size, Image.BICUBIC)

    padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - resized_size[0]) // 2
    paste_y = (target_size - resized_size[1]) // 2
    padded_image.paste(image, (paste_x, paste_y))

    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_pipeline(padded_image).unsqueeze(0).to(DEVICE)


def get_model(model_name: str, num_classes: int = 7):
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {model_name}: {exc}") from exc


def run_inference(model, image_tensor):
    with torch.no_grad():
        tensor_input = image_tensor.to(DEVICE, non_blocking=True)
        outputs = model(tensor_input)
        return torch.softmax(outputs, dim=1)


def predict_fast(image_path: str, metadata: Optional[Dict[str, Any]] = None) -> dict:
    validation = validate_image_path(image_path)
    if validation["status"] != "ok":
        return validation

    image_tensor = process_image(image_path, target_size=224)
    model = get_model("efficientnet_b0", num_classes=7)
    probabilities = run_inference(model, image_tensor)

    confidence, class_idx = probabilities.max(dim=1)
    idx = class_idx.item()

    result = {
        "status": "success",
        "tool": "skin-lesion-fast",
        "model_tier": "tier1_fast",
        "model_executed": "efficientnet_b0",
        "predicted_class_index": idx,
        "disease_name": LABEL_MAP.get(idx, "Unknown Pathological Condition"),
        "confidence_score": round(confidence.item(), 4)
    }

    if metadata is not None:
        result["metadata"] = metadata

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/skin_lesion_fast.py",
        description="Run the fast skin lesion screening model using EfficientNet-B0."
    )
    parser.add_argument("--image", dest="image_path", required=True, help="Path to the lesion image.")
    parser.add_argument("--metadata", dest="metadata", required=False, help="Optional JSON metadata about the patient or image.")
    args = parser.parse_args()

    metadata = None
    try:
        metadata = parse_metadata(args.metadata)
    except ValueError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, ensure_ascii=False))
        return 1

    result = predict_fast(args.image_path, metadata=metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import ast
import json
import re
import sys
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

SUPPORTED_CUDA_SM = {
    (5, 0),
    (6, 0),
    (6, 1),
    (7, 0),
    (7, 5),
    (8, 0),
    (8, 6),
    (9, 0),
}

DEVICE = torch.device("cpu")


def select_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available: no CUDA-compatible device was found. "
            "Install a CUDA-capable PyTorch build or run on a supported GPU."
        )

    try:
        capability = torch.cuda.get_device_capability()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to determine CUDA device capability: {exc}"
        ) from exc

    if capability not in SUPPORTED_CUDA_SM:
        raise RuntimeError(
            f"CUDA device compute capability sm_{capability[0]}{capability[1]} "
            "is not supported by the installed PyTorch build."
        )

    return torch.device("cuda")


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

    def normalize_unquoted(obj_str: str) -> str:
        obj_str = obj_str.strip()
        if not obj_str.startswith("{") or not obj_str.endswith("}"):
            return obj_str

        obj_str = re.sub(r'(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', obj_str)

        def quote_value(match: re.Match) -> str:
            value = match.group(1)
            if re.fullmatch(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', value):
                return ":" + value
            if value.lower() in {"true", "false", "null"}:
                return ":" + value.lower()
            return ':"' + value.replace('"', '\\"') + '"'

        return re.sub(
            r':\s*([A-Za-z_][A-Za-z0-9_]*)(?=\s*(?:,|\}))',
            quote_value,
            obj_str,
        )

    for candidate in [cleaned, cleaned.replace("'", '"'), normalize_unquoted(cleaned)]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    raise ValueError(
        f"Invalid metadata JSON: Received: {metadata_json}"
    )


def process_image(image_path: str, target_size: int = 380):
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


def predict_mid(image_path: str, metadata: Optional[Dict[str, Any]] = None) -> dict:
    validation = validate_image_path(image_path)
    if validation["status"] != "ok":
        return validation

    image_tensor = process_image(image_path, target_size=380)
    model = get_model("efficientnet_b4", num_classes=7)
    probabilities = run_inference(model, image_tensor)

    confidence, class_idx = probabilities.max(dim=1)
    idx = class_idx.item()

    result = {
        "status": "success",
        "tool": "skin-lesion-mid",
        "model_tier": "tier2_mid",
        "model_executed": "efficientnet_b4",
        "predicted_class_index": idx,
        "disease_name": LABEL_MAP.get(idx, "Unknown Pathological Condition"),
        "confidence_score": round(confidence.item(), 4)
    }

    if metadata is not None:
        result["metadata"] = metadata

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/skin_lesion_mid.py",
        description="Run the mid-tier skin lesion screening model using EfficientNet-B4."
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

    global DEVICE
    try:
        DEVICE = select_cuda_device()
        torch.backends.cudnn.benchmark = True
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, ensure_ascii=False))
        return 1

    result = predict_mid(args.image_path, metadata=metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

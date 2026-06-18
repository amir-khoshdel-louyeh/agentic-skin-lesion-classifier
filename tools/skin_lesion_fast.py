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
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}

# به‌روزرسانی شده جهت پشتیبانی از معماری کارت گرافیک شما (sm_120)
SUPPORTED_CUDA_SM = {
    (5, 0),
    (6, 0),
    (6, 1),
    (7, 0),
    (7, 5),
    (8, 0),
    (8, 6),
    (9, 0),
    (12, 0),
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


def process_image(image_path: str, target_size: int = 224):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    # تغییر سایز استاندارد
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_size = (int(width * scale), int(height * scale))
    image = image.resize(resized_size, Image.BICUBIC)

    padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - resized_size[0]) // 2
    paste_y = (target_size - resized_size[1]) // 2
    padded_image.paste(image, (paste_x, paste_y))

    # تبدیل به BGR جهت هماهنگی کامل با دیتای آموزشی مدل شما
    r, g, b = padded_image.split()
    bgr_image = Image.merge("RGB", (b, g, r))

    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        # نرمال‌سازی استاندارد متناسب با پایپ‌لاین‌های BGR در PyTorch
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    return transform_pipeline(bgr_image).unsqueeze(0).to(DEVICE)


def get_model(model_name: str, num_classes: int = 7) -> torch.nn.Module:
    """
    Initializes a raw model architecture from the timm library and loads
    the custom local fine-tuned weights for skin lesion classification.
    """
    try:
        # 1. Initialize the raw model structure without ImageNet default weights
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # 2. Define the path to your verified HAM10000 weights file
        weights_path = "ham10000_efficientnet_b0.pth"
        
        # 3. Load the checkpoint onto the correct execution device
        state_dict = torch.load(weights_path, map_location=DEVICE)
        
        # 4. Handle cases where the weights are nested inside an outer dictionary wrapper
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # 5. Inject the custom fine-tuned weights into the architecture
        # استفاده از strict=False برای جلوگیری از خطاهای احتمالی ناشی از تفاوت جزئی نام‌گذاری لایه‌ها
        model.load_state_dict(state_dict, strict=False)
        
        # 6. Push model parameters to target device (CPU/CUDA) and switch to evaluation mode
        model = model.to(DEVICE)
        model.eval()
        
        return model
        
    except FileNotFoundError as fnf_exc:
        raise RuntimeError(
            f"The fine-tuned weights file was not found at '{weights_path}'. "
            "Please ensure the file is placed in the correct directory."
        ) from fnf_exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load the fine-tuned model '{model_name}' using weights from '{weights_path}': {exc}"
        ) from exc


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

    global DEVICE
    try:
        DEVICE = select_cuda_device()
        torch.backends.cudnn.benchmark = True
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, ensure_ascii=False))
        return 1

    result = predict_fast(args.image_path, metadata=metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
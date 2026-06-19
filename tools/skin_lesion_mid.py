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
import torchvision.models as models
from torchvision import transforms
from huggingface_hub import hf_hub_download

# ترتیب حروف الفبایی رسمی دیتاست HAM10000 هماهنگ با مدل DermAI
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

SUPPORTED_CUDA_SM = {
    (5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (12, 0)
}

DEVICE = torch.device("cpu")


def select_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available.")
    try:
        capability = torch.cuda.get_device_capability()
    except Exception as exc:
        raise RuntimeError(f"Unable to determine CUDA capability: {exc}") from exc
    if capability not in SUPPORTED_CUDA_SM:
        raise RuntimeError(f"CUDA device sm_{capability[0]}{capability[1]} not supported.")
    return torch.device("cuda")


def validate_image_path(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return {"status": "error", "message": f"Invalid image path: {image_path}"}
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return {"status": "error", "message": "Unsupported file type."}
    return {"status": "ok"}


def parse_metadata(metadata_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not metadata_json:
        return None
    cleaned = metadata_json.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "'":
        cleaned = cleaned[1:-1].strip()
    try:
        return json.loads(cleaned.replace("'", '"'))
    except Exception:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            raise ValueError(f"Invalid metadata JSON: {metadata_json}")


def get_model() -> torch.nn.Module:
    """
    Initializes a native torchvision ConvNeXt-Tiny architecture to precisely
    match the state_dict keys of the DermAI checkpoint.
    """
    repo_id = "imtiazhumzah/DermAI-Clinical-Screen"
    filename = "best_melanoma_recall_model.pth"
    
    try:
        # ۱. ساخت ساختار خام معماری سازگار با فایل وزن‌ها از طریق torchvision
        # در ساختار نیتیو، لایه نهایی کلاسیفایر خود به خود روی ۱۰۰۰ تنظیم است.
        model = models.convnext_tiny(pretrained=False)
        
        # ۲. دانلود خودکار وزن‌ها از هاگینگ فیس
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # ۳. لود کردن وزن‌ها (اکنون کلیدها کاملاً تطابق دارند اما لایه کلاسیفایر خطا خواهد داد چون مال مدل ۷ تایی است)
        # برای رفع این مشکل، موقتاً لایه آخر مدل خام را تغییر می‌دهیم تا با لایه آخر ذخیره شده سازگار شود
        # ساختار لایه آخر در torchvision convnext به صورت یک لایه Sequential با دو بخش خطی است:
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 7)
        
        # ۴. حالا تزریق وزن‌ها بدون کوچک‌ترین خطایی انجام می‌شود
        model.load_state_dict(state_dict, strict=True)
        
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load DermAI model/weights: {exc}") from exc
    


def predict_mid(image_path: str, metadata: Optional[Dict[str, Any]] = None) -> dict:
    validation = validate_image_path(image_path)
    if validation["status"] != "ok":
        return validation

    # ۱. لود مدل DermAI
    model = get_model()

    # ۲. پیش‌پردازش استاندارد با مقادیر ImageNet برای رزولوشن ۲۲۴ (جایگزین تمپلیت قبلی timm)
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ۳. لود تصویر
    raw_image = Image.open(image_path)
    raw_image = ImageOps.exif_transpose(raw_image).convert("RGB")
    image_tensor = transform_pipeline(raw_image).unsqueeze(0).to(DEVICE)

    # ۴. اجرای اینفرنس
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, class_idx = probabilities.max(dim=1)
    idx = class_idx.item()

    result = {
        "status": "success",
        "tool": "skin-lesion-mid",
        "model_tier": "tier2_mid",
        "model_executed": "convnext_tiny_dermai",
        "predicted_class_index": idx,
        "disease_name": LABEL_MAP.get(idx, "Unknown Condition"),
        "confidence_score": round(confidence.item(), 4)
    }

    if metadata is not None:
        result["metadata"] = metadata

    return result



def main() -> int:
    parser = argparse.ArgumentParser(prog="python tools/skin_lesion_mid.py")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    metadata = None
    try:
        metadata = parse_metadata(args.metadata)
    except ValueError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1

    global DEVICE
    try:
        DEVICE = select_cuda_device()
        torch.backends.cudnn.benchmark = True
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1

    result = predict_mid(args.image_path, metadata=metadata)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
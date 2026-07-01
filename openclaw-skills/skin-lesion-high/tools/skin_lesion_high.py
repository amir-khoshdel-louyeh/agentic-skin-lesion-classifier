import argparse
import json
import os
import sys
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms

# لایبل‌های استاندارد ۷ کلاسی هماهنگ با مدل HAM10000
CLASSES = [
    "Actinic keratoses",        # 0
    "Basal cell carcinoma",     # 1
    "Benign keratosis",         # 2
    "Dermatofibroma",           # 3
    "Melanoma",                 # 4
    "Melanocytic nevi",         # 5
    "Vascular lesions"          # 6
]

SUPPORTED_CUDA_SM = {(5,0), (6,0), (6,1), (7,0), (7,5), (8,0), (8,6), (9,0), (12,0)}
DEVICE = torch.device("cpu")

def select_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        capability = torch.cuda.get_device_capability()
        if capability in SUPPORTED_CUDA_SM:
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

def get_high_tier_model(device):
    print("⏳ Loading Offline SOTA ViT-Large from models/offline_high...", file=sys.stderr)
    from transformers import AutoModelForImageClassification
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_path = os.path.join(BASE_DIR, "models", "offline_high")
    
    if not os.path.exists(local_model_path):
        print(json.dumps({"status": "error", "message": f"Local high model folder missing at: {local_model_path}"}))
        sys.exit(1)
        
    model = AutoModelForImageClassification.from_pretrained(local_model_path, local_files_only=True)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="SOTA ViT-Large Offline High-Tier Classifier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    global DEVICE
    DEVICE = select_cuda_device()

    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error", "message": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    # پردازش ایمن متادیتا (سازگار با JSONL)
    metadata_json = {}
    if args.metadata:
        try:
            metadata_json = json.loads(args.metadata)
        except json.JSONDecodeError:
            metadata_json = {"error": "Invalid metadata format"}

    # پایپ‌لاین پیش‌پردازش
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        raw_image = Image.open(args.image_path)
        raw_image = ImageOps.exif_transpose(raw_image).convert("RGB")
        image_tensor = transform_pipeline(raw_image).unsqueeze(0).to(DEVICE)

        model = get_high_tier_model(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_idx = torch.max(probabilities, dim=0)

        idx = int(class_idx.item())

        result = {
            "status": "success",
            "tool": "skin-lesion-high",
            "model_tier": "tier3_high",
            "model_executed": "vit_large_ham10000_offline",
            "predicted_class_index": idx,
            "disease_name": CLASSES[idx] if idx < len(CLASSES) else "Unknown Condition",
            "confidence_score": round(float(confidence.item()), 4),
            "metadata": metadata_json
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Execution failed: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
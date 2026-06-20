import argparse
import json
import os
import sys
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor

# لایبل‌های استاندارد ۷ کلاسی
CLASSES = [
    "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis", 
    "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"
]

SUPPORTED_CUDA_SM = {(5,0), (6,0), (6,1), (7,0), (7,5), (8,0), (8,6), (9,0), (12,0)}
DEVICE = torch.device("cpu")

def select_cuda_device() -> torch.device:
    if not torch.cuda.is_available(): return torch.device("cpu")
    try:
        capability = torch.cuda.get_device_capability()
        if capability in SUPPORTED_CUDA_SM: return torch.device("cuda")
    except Exception: pass
    return torch.device("cpu")

def get_fast_tier_model(device):
    print("⏳ Loading Offline Fast-Tier ConvNeXt with manual processor...", file=sys.stderr)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_path = os.path.join(BASE_DIR, "models", "offline_fast")
    
    # لود مدل
    model = AutoModelForImageClassification.from_pretrained(local_model_path, local_files_only=True)
    
    # لود دستی پردازشگر (استفاده از تنظیمات پیش‌فرض ConvNeXt)
    # مدل‌های ConvNeXt معمولاً از سایز 224 استفاده می‌کنند
    from transformers import ConvNextImageProcessor
    processor = ConvNextImageProcessor.from_pretrained(local_model_path, local_files_only=True)
    
    model.to(device)
    model.eval()
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="Fast-Tier ConvNeXt Offline Structural Classifier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    global DEVICE
    DEVICE = select_cuda_device()

    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error", "message": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    try:
        raw_image = Image.open(args.image_path)
        raw_image = ImageOps.exif_transpose(raw_image).convert("RGB")
        
        # لود مدل و پردازشگر
        model, processor = get_fast_tier_model(DEVICE)
        
        # پیش‌پردازش خودکار متناسب با معماری ConvNeXt
        inputs = processor(images=raw_image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_idx = torch.max(probabilities, dim=0)

        idx = int(class_idx.item())

        metadata_json = {}
        if args.metadata:
            try:
                metadata_json = json.loads(args.metadata.replace("'", '"'))
            except Exception: pass

        result = {
            "status": "success",
            "tool": "skin-lesion-fast",
            "model_tier": "tier1_fast",
            "model_executed": "convnext_small_skin_lesion_offline",
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
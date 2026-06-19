import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# برگشتن به ساختار ۷ کلاسی منطبق با لایه‌های قبلی و فایل وزن شما
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

def select_cuda_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        capability = torch.cuda.get_device_capability()
        if capability in SUPPORTED_CUDA_SM:
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

def get_model(model_path, device):
    import torchvision.models as models
    
    # بازسازی بدنه مدل بر اساس معماری وزنی ۱۱۱ مگابایتی
    model = models.convnext_tiny(weights=None)
    num_ftrs = model.classifier[2].in_features
    
    # تنظیم دقیق روی لایه خروجی ۷ کلاسی جهت حل مشکل Size Mismatch
    model.classifier[2] = nn.Linear(num_ftrs, len(CLASSES))
    
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    fixed_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("model."):
            new_key = k.replace("model.", "")
        elif k.startswith("module."):
            new_key = k.replace("module.", "")
        fixed_state_dict[new_key] = v

    # استفاده از strict=True چون حالا ابعاد ماتریس‌ها کاملاً با وزن‌ها جفت شده‌اند
    model.load_state_dict(fixed_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="High-tier accurate vision classifier tool for OpenClaw.")
    parser.add_argument("--image", required=True, help="Path to the lesion image")
    parser.add_argument("--metadata", required=False, default="{}")
    args = parser.parse_args()

    global DEVICE
    DEVICE = select_cuda_device()

    if not os.path.exists(args.image):
        print(json.dumps({"status": "error", "message": f"Image not found at: {args.image}"}))
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(args.image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_melanoma_recall_model.pth")

        if not os.path.exists(model_path):
            print(json.dumps({"status": "error", "message": "High tier weights file (.pth) missing inside tools/"}))
            sys.exit(1)

        model = get_model(model_path, DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, dim=0)

        idx = int(pred_idx.item())
        result = {
            "status": "success",
            "tool": "skin-lesion-high",
            "model_tier": "tier3_high",
            "model_executed": "convnext_high_recall",
            "predicted_class_index": idx,
            "disease_name": CLASSES[idx] if idx < len(CLASSES) else "Unknown Condition",
            "confidence_score": round(float(conf.item()), 4),
            "metadata": json.loads(args.metadata)
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
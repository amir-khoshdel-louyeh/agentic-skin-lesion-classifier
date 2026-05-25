# core/__init__.py
import torch
from core.model_factory import ModelFactory
from core.processors import ImageProcessor

# نگاشت شناسه‌های عددی مجموعه داده به نام‌های واقعی استاندارد پزشکی
LABEL_MAP = {
    0: "Actinic keratoses (ضایعات پیش‌سرطانی)",
    1: "Basal cell carcinoma (سرطان سلول پایه‌ای)",
    2: "Benign keratosis (کراتوز خوش‌خیم پوست)",
    3: "Dermatofibroma (درماتوفیبروما)",
    4: "Melanocytic nevi (خال‌های خوش‌خیم)",
    5: "Melanoma (ملانوما - سرطان بدخیم پوست)",
    6: "Vascular lesions (ضایعات عروقی)"
}

def analyze_skin_lesion(image_path: str, model_tier: str) -> dict:
    """
    An advanced clinical tool to analyze skin lesions using deep learning models.
    
    Parameters:
    - image_path (str): The local path to the patient's lesion image.
    - model_tier (str): Must be either 'tier1_fast' (uses Lightweight EfficientNet-B0) 
                        or 'tier2_deep' (uses High-Accuracy EfficientNet-B4).
    """
    try:
        model_tier = model_tier.lower()
        
        # ۱. مدیریت پنهان تغییر سایز و انتخاب مدل بر اساس انتخاب لایه توسط عامل
        if model_tier == "tier1_fast":
            target_size = 224
            model_name = "efficientnet_b0"
        elif model_tier == "tier2_deep":
            target_size = 380
            model_name = "efficientnet_b4"
        else:
            return {
                "status": "error", 
                "message": "Invalid model_tier. You must explicitly choose 'tier1_fast' or 'tier2_deep'."
            }
        
        print(f"[Internal Tool] Processing: Resizing image to {target_size}x{target_size} for {model_name}...")
        
        # ۲. پیش‌پردازش تصویر
        image_tensor = ImageProcessor.process_image(image_path, target_size=target_size)
        
        # ۳. دریافت مدل و اجرای استنتاج
        model = ModelFactory.get_model(model_name, num_classes=7)
        probabilities = ModelFactory.run_inference(model, image_tensor)
        
        # ۴. استخراج بالاترین احتمال و ایندکس مربوطه
        confidence, class_idx = probabilities.max(dim=1)
        idx = class_idx.item()
        
        return {
            "status": "success",
            "model_executed": model_name,
            "predicted_class_index": idx,
            "disease_name": LABEL_MAP.get(idx, "Unknown Pathological Condition"),
            "confidence_score": round(confidence.item(), 4)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Execution failed: {str(e)}"}
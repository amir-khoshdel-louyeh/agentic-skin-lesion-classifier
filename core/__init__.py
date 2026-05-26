# core/__init__.py
import torch
from pathlib import Path
from PIL import Image
from core.model_factory import ModelFactory
from core.processors import ImageProcessor

# Map numeric dataset labels to standard medical names
LABEL_MAP = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanocytic nevi",
    5: "Melanoma",
    6: "Vascular lesions"
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def validate_image_input(image_path: str) -> dict:
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
            "message": f"Unsupported image type: {path.suffix}. Supported types: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        }

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Bad input image before inference: {str(exc)}"
        }

    return {"status": "ok"}


def analyze_skin_lesion(image_path: str, model_tier: str) -> dict:
    """
    An advanced clinical tool to analyze skin lesions using deep learning models.
    
    Parameters:
    - image_path (str): The local path to the patient's lesion image.
    - model_tier (str): Must be either 'tier1_fast' (uses Lightweight EfficientNet-B0) 
                        or 'tier2_deep' (uses High-Accuracy EfficientNet-B4).
    """
    try:
        validation = validate_image_input(image_path)
        if validation["status"] != "ok":
            return validation

        model_tier = model_tier.lower()
        
        # 1. Choose the correct model and input size for the selected tier
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
        
        # 2. Preprocess the image
        image_tensor = ImageProcessor.process_image(image_path, target_size=target_size)
        
        # 3. Load the model and run inference
        model = ModelFactory.get_model(model_name, num_classes=7)
        probabilities = ModelFactory.run_inference(model, image_tensor)
        
        # 4. Extract the highest probability and predicted class index
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
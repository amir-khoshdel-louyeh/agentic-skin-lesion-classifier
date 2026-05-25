# core/processors.py
import os
from PIL import Image
from torchvision import transforms

class ImageProcessor:
    @staticmethod
    def process_image(image_path: str, target_size: int = 224):
        """
        opens an image path, resizes it dynamically according to the model needs,
        and converts it to a PyTorch tensor with ImageNet normalization.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        # باز کردن تصویر
        image = Image.open(image_path).convert('RGB')
        
        # خط لوله پردازش متناسب با مدل‌های Pre-trained
        transform_pipeline = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # تبدیل به تنسور و افزودن بعد Batch (اندازه خروجی: [1, 3, target_size, target_size])
        return transform_pipeline(image).unsqueeze(0)
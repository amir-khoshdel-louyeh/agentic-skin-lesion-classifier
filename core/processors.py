# core/processors.py
import os
from PIL import Image, ImageOps
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
            
        # باز کردن تصویر و تصحیح جهت بر اساس EXIF
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert('RGB')

        # حفظ نسبت تصویر هنگام تغییر اندازه و اضافه کردن پد به یک ورودی مربع
        width, height = image.size
        scale = min(target_size / width, target_size / height)
        resized_size = (int(width * scale), int(height * scale))
        image = image.resize(resized_size, Image.BICUBIC)

        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - resized_size[0]) // 2
        paste_y = (target_size - resized_size[1]) // 2
        padded_image.paste(image, (paste_x, paste_y))

        # خط لوله پردازش متناسب با مدل‌های Pre-trained
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # تبدیل به تنسور و افزودن بعد Batch (اندازه خروجی: [1, 3, target_size, target_size])
        return transform_pipeline(padded_image).unsqueeze(0)
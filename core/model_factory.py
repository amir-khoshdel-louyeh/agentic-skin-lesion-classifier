# core/model_factory.py
import torch
import timm

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, num_classes: int = 7):
        """
        Dynamically downloads and configures pretrained CNN architectures.
        Sets the classifier head to match num_classes and moves to GPU if available.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = model_name.lower()
        
        try:
            # ایجاد مدل آماده به همراه بازنویسی خودکار لایه نهایی بر اساس کلاس‌های ما
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            model = model.to(device)
            model.eval()  # غیرفعال کردن Dropout برای استنتاج پایدار
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name} via timm. Error: {str(e)}")

    @staticmethod
    def run_inference(model, image_tensor):
        """
        Helper method to run a safe inference without calculating gradients.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            tensor_input = image_tensor.to(device)
            outputs = model(tensor_input)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities
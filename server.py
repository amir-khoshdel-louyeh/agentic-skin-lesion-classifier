from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = FastAPI(title="CNN Ensemble Server")

# ----------------------------
# Dummy models (replace later with real ones)
# ----------------------------
class DummyModel:
    def __init__(self, name):
        self.name = name

    def predict(self, x):
        # fake probabilities (replace with real model output)
        torch.manual_seed(hash(self.name) % 1000)
        return torch.softmax(torch.randn(4), dim=0)

resnet = DummyModel("resnet")
effnet = DummyModel("efficientnet")
dense = DummyModel("densenet")

models = [resnet, effnet, dense]

classes = ["akiec", "bcc", "melanoma", "nv"]

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess(image: Image.Image):
    return transform(image).unsqueeze(0)

# ----------------------------
# Ensemble logic
# ----------------------------
def ensemble(preds):
    return torch.stack(preds).mean(dim=0)

# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = preprocess(image)

    preds = []
    for model in models:
        preds.append(model.predict(x))

    final = ensemble(preds)

    result = {
        "predictions": {
            classes[i]: float(final[i])
            for i in range(len(classes))
        },
        "final_class": classes[int(torch.argmax(final))]
    }

    return result
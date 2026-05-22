#FastAPI CNN Service


""" EXAMPLE: 


from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
from PIL import Image
import io

app = FastAPI()

# =========================
# Load your CNN model here
# =========================
model = None  # load your pretrained model here


def load_model():
    global model
    # مثال:
    # model = torch.load("models/cnn.pt", map_location="cpu")
    # model.eval()
    pass


def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # TODO: resize / normalize
    return image


@app.on_event("startup")
def startup():
    load_model()
    print("✅ CNN model loaded")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = preprocess(image_bytes)

    # =========================
    # inference
    # =========================
    # pred = model(image)
    # class_name = decode(pred)

    # mock output for now:
    result = {
        "class": "melanoma",
        "confidence": 0.94
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

    """
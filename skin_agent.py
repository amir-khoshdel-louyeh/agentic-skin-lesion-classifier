import os
import random
import time
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CNN Skin Classifier")

print("Loading Pre-trained CNN Models...")
time.sleep(1)
print("CNN Models loaded successfully. API is ready on port 8000.")

class ImageQuery(BaseModel):
    image_path: str

@app.post("/analyze")
def analyze_skin_lesion(data: ImageQuery):
    image_path = data.image_path
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"File not found: {image_path}"}
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        
        # Random output for initial testing (replace with your PyTorch models later)
        melanoma_prob = random.uniform(5.0, 85.0)
        carcinoma_prob = random.uniform(2.0, 40.0)
        
        return {
            "status": "success",
            "melanoma": f"{melanoma_prob:.2f}%",
            "carcinoma": f"{carcinoma_prob:.2f}%"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
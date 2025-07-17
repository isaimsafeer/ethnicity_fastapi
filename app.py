from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np
import cv2
import io
from pathlib import Path
from model import EthnicityModel
from inference import classify_ethnicity
from datetime import datetime
import os

app = FastAPI()

# Load model

base_dir = Path(__file__).resolve().parent

# Construct the path to the data folder inside your project folder
MODEL_PATH = base_dir/ "Model" / "refined_v1_epoch_22.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ethnicity_labels = {
    0: "Black",
    1: "South East Asian",
    2: "Indian",
    3: "White",
}

num_classes = len(ethnicity_labels)
model = EthnicityModel(num_classes)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Ensure the output directory exists
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/predict/")
async def predict_ethnicity(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Read image into OpenCV format
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Generate unique filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(SAVE_DIR, filename)

    # Save image
    cv2.imwrite(save_path, image)

    # Run prediction
    result = classify_ethnicity(save_path, model, ethnicity_labels, device=device, save_dir=SAVE_DIR)


    if isinstance(result, dict):
        return JSONResponse(content={
            "ethnicity": result["ethnicity"],
            "confidence": result["confidence"],
            "face_confidence": result["face_confidence"],
            "all_probabilities": result.get("all_probabilities", {}),
            "saved_path": save_path
        })
    else:
        raise HTTPException(status_code=400, detail=result)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import cv2
import numpy as np
import shutil
import tempfile
import os
import uvicorn

import train     # your train.py
import predict   # your predict.py

# 1) Create the FastAPI app
app = FastAPI(title="Packaging Detector API")

# 2) Load the model once at startup
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "/Users/allenpereira/Desktop/d33ewd 2/runs/detect/new_run19/weights/best.pt")
model = predict.load_model(MODEL_WEIGHTS)

# 3) Multipart/form-data file upload endpoint
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...), conf: float = 0.5):
    # Save upload to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        results = predict.predict_on_image(model, tmp_path, conf_threshold=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return JSONResponse(results)

# 4) Pydantic model for URL-based prediction
class PredictURL(BaseModel):
    url: str
    conf: float = 0.5

# 5) JSON payload endpoint for URLs
@app.post("/predict_url/")
async def predict_url_endpoint(payload: PredictURL):
    # 5.1) Download image bytes
    resp = requests.get(payload.url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {resp.status_code}")

    # 5.2) Decode to OpenCV image
    img_arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image from URL")

    # 5.3) Run prediction on the array
    results = predict.predict_on_image(model, img, conf_threshold=payload.conf)
    return JSONResponse(results)

# 6) Training endpoint
@app.post("/train/")
async def train_endpoint(epochs: int = 50, batch: int = 10, lr0: float = 0.01):
    train.remove_old_run("api_run")
    train.train_yolo(
        data_yaml="packaging_data.yaml",
        model_name="yolov8n.pt",
        epochs=epochs,
        run_name="api_run",
        batch=batch,
        lr0=lr0
    )
    return {"status": "training_started", "epochs": epochs}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import tempfile


import train   
import predict 

app = FastAPI(title="Packaging Detector API")

# Load or initialize model at startup
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "/Users/allenpereira/Desktop/d33ewd 2/runs/detect/new_run19/weights/best.pt")
model = predict.load_model(MODEL_WEIGHTS)

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...), conf: float = 0.5):
    # Save upload to a temporary file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        results = predict.predict_on_image(model, tmp_path, conf_threshold=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return JSONResponse(results)

@app.post("/train/")
async def train_endpoint(epochs: int = 50, batch: int = 10, lr0: float = 0.01):
    # Remove old run
    train.remove_old_run("api_run")
    # Kick off training
    model = train.train_yolo(
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

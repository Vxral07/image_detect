import os
import shutil
import cv2
import numpy as np
import random
import logging
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode as decode_barcode, ZBarSymbol

# ---------------------------
# Setup Logging and Seeds
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

# ---------------------------
# Utility Functions
# ---------------------------
def create_default_yaml(file_path="packaging_data.yaml"):
    """
    Creates a default YAML configuration file if one doesn't exist.
    Assumes:
      - Training images are in /Users/allenpereira/Desktop/d33ewd 2/datasets/images/train
      - Validation images are in /Users/allenpereira/Desktop/d33ewd 2/datasets/images/val
      - Classes: barcode, brand, price, description, product_name
    """
    default_yaml = """\
train: /Users/allenpereira/Desktop/image_git/datasets/images/train
val: /Users/allenpereira/Desktop/image_git/datasets/images/val
names:
  0: barcode
  1: brand
  2: price
  3: description
  4: product_name
"""
    with open(file_path, "w") as f:
        f.write(default_yaml)
    logging.info(f"Created default YAML configuration at '{file_path}'.")

def remove_old_run(run_name="packaging-detector"):
    """
    Removes the old training run folder if it exists.
    Prevents resuming from previous (possibly bad) checkpoints.
    """
    run_dir = os.path.join("runs", "detect", run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        logging.info(f"Removed old run directory: {run_dir}")
    else:
        logging.info(f"No old run directory found at: {run_dir}")

# ---------------------------
# Rotated Box Conversion Function
# ---------------------------
def rotated_box_to_axis_aligned(center_x, center_y, width, height, angle):
    """
    Converts a rotated bounding box (in pixel coordinates) defined by its center,
    width, height, and angle into an axis-aligned bounding box.
    
    Returns:
        list: [x1, y1, x2, y2] in pixel coordinates.
    """
    rect = ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)
    return [x, y, x + w, y + h]

# ---------------------------
# Preprocessing Functions
# ---------------------------
def unblur_image(image):
    """
     reduce blurriness and enhance edges.
    """
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return unsharp

# ---------------------------
# OCR and Barcode Helper Functions
# ---------------------------
def crop_region(image, box):
    """
    Crop a region from the image based on YOLO's bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def ocr_read_text(cropped_img, reader, scale_factor=1.0):
    """
    Use PaddleOCR to detect and recognize text from a cropped region.
    """
    if scale_factor > 1.0:
        cropped_img = cv2.resize(cropped_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    result = reader.ocr(cropped_rgb, cls=True)
    texts = []
    if result is not None:
        for line in result:
            if line is not None:
                for detection in line:
                    if detection and detection[1] and len(detection[1]) >= 1:
                        texts.append(detection[1][0])
    return " ".join(texts)

def decode_barcode_image(cropped_img):
    """
    Attempt to decode barcode(s) in the cropped region using pyzbar.
    """
    decoded_objects = decode_barcode(cropped_img, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE128,
                                                           ZBarSymbol.QRCODE, ZBarSymbol.EAN8])
    if decoded_objects:
        return decoded_objects[0].data.decode("utf-8")
    return ""

def sample_inference(model, image_path, conf_threshold=0.5):
    """
    Runs inference on an image, processing each detected region with OCR/barcode decoding.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image from {image_path}")
    original_img = unblur_image(original_img)
    
    results = model.predict(source=original_img, conf=conf_threshold)
    detections = results[0].boxes

    # Order of classes must match your YAML.
    CLASS_NAMES = ["brand", "product_name", "price", "description", "barcode"]
    output_data = {cls: [] for cls in CLASS_NAMES}
    
    # Instantiate PaddleOCR (runs on CPU for Mac M1)
    ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

    if len(detections) > 0:
        for det in detections:
            box = det.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            cls_id = int(det.cls[0].cpu().numpy())
            cls_name = CLASS_NAMES[cls_id]
            cropped = crop_region(original_img, box)
            
            h, w, _ = cropped.shape
            scale = 1.0
            if w < 50 or h < 50:
                scale = 2.0
            if cls_name in ["brand", "product_name"]:
                scale = max(scale, 2.0)
            
            if cls_name == "barcode":
                barcode_value = decode_barcode_image(cropped)
                if not barcode_value or not barcode_value.strip():
                    logging.info("Barcode data is unreadable; using OCR for barcode region.")
                    barcode_value = ocr_read_text(cropped, ocr_reader, scale_factor=1.0)
                    if not barcode_value or not barcode_value.strip():
                        logging.info("Fallback OCR failed for barcode region.")
                        barcode_value = "Barcode unreadable"
                output_data["barcode"].append(barcode_value)
            else:
                text = ocr_read_text(cropped, ocr_reader, scale_factor=scale)
                output_data[cls_name].append(text)
    else:
        logging.info("No detections found by YOLO.")
    
    logging.info("Running full-image OCR for additional text detection.")
    full_text = ocr_read_text(original_img, ocr_reader, scale_factor=1.0)
    output_data["all_text"] = full_text

    return output_data

# ---------------------------
# Training Function with Exposed Hyperparameters
# ---------------------------
def train_yolo(data_yaml: str,
               model_name: str = "yolov8n.pt",
               epochs: int = 40,
               run_name: str = "new_run",
               imgsz: int = 900,
               batch: int = 10,
               lr0: float = 0.01,
               lrf: float = 0.01):
    """
    Trains a YOLO model using the specified dataset configuration.
    """
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,         # initial learning rate
        lrf=lrf,         # final learning rate factor
        patience=10,
        augment=True,    # Data augmentation (rotation, flip, etc.)
        name=run_name
    )
    return model

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    set_random_seed(42)
    
    data_yaml_path = "packaging_data.yaml"
    if not os.path.exists(data_yaml_path):
        create_default_yaml(data_yaml_path)
    
    remove_old_run("packaging-detector")
    
    # Train the model with updated parameters.
    model = train_yolo(data_yaml=data_yaml_path,
                       model_name="yolov8n.pt",
                       epochs=40,
                       run_name="new_run",
                       imgsz=900,
                       batch=10,
                       lr0=0.01,
                       lrf=0.01)
    
    # --- Sample Inference ---
    sample_image_path = "/Users/allenpereira/Desktop/image_git/datasets/images/train/0a1247c1-test100.jpeg"
    
    try:
        inference_results = sample_inference(model, sample_image_path)
        logging.info("Sample Inference Output:")
        logging.info(inference_results)
    except Exception as e:
        logging.error(f"Sample inference failed: {e}")

    logging.info("Training complete. rn storing in runs folder")

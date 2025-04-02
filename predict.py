import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode as decode_barcode, ZBarSymbol

# Class order og yaml
CLASS_NAMES = [
    "barcode",
    "brand",
    "price",
    "description",
    "product_name"
]

# Parameters for scaling
SMALL_REGION_THRESHOLD = 50   # pixels; if region is smaller, scale up
DEFAULT_SCALE_FACTOR = 1.0
MIN_SCALE_BRAND_PRODUCT = 2.0   # For "brand" and "product_name", enforce a minimum scale

def load_model(weights_path: str):
    """
    Load the YOLO model from the given weights file.
    """
    model = YOLO(weights_path)
    return model

def crop_region(image, box):
    """
    Crop a region from the image based on YOLO's bounding box coordinates [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def unblur_image(image):
    """
    Apply an unsharp mask to the image to reduce blurriness and enhance edges.
    """
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return unsharp

def ocr_read_text(cropped_img, reader, scale_factor=1.0):
    """
    Use PaddleOCR to extract text from a cropped region.
    Optionally scales up the image for better OCR of small text.
    Converts the image from BGR to RGB.
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
    Attempt to decode barcode(s) from a cropped region using pyzbar.
    """
    decoded_objects = decode_barcode(cropped_img, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE128,
                                                           ZBarSymbol.QRCODE, ZBarSymbol.EAN8])
    if decoded_objects:
        return decoded_objects[0].data.decode("utf-8")
    return ""

def predict_on_image(model, image_path, conf_threshold=0.5):
    """
    Run YOLO inference on an image, process detections using the same preprocessing 
    and scaling logic as in training, and return a dictionary with the extracted text.
    A full-image OCR is also run as a backup.
    """
    # Read the input image and apply unblurring.
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image from {image_path}")
    original_img = unblur_image(original_img)
    
    # Run YOLO inference.
    results = model.predict(source=original_img, conf=conf_threshold)
    detections = results[0].boxes

    output_data = {
        "barcode": [],
        "brand": [],
        "price": [],
        "description": [],
        "product_name": []
    }
    
    # Instantiate PaddleOCR (runs on CPU on your Mac M1 Air)
    ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

    if len(detections) > 0:
        for det in detections:
            box = det.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2]
            cls_id = int(det.cls[0].cpu().numpy())
            cls_name = CLASS_NAMES[cls_id]
            cropped = crop_region(original_img, box)
            
            # Determine scaling factor based on region size.
            h, w, _ = cropped.shape
            scale = DEFAULT_SCALE_FACTOR
            if w < SMALL_REGION_THRESHOLD or h < SMALL_REGION_THRESHOLD:
                scale = 2.0
            if cls_name in ["brand", "product_name"]:
                scale = max(scale, MIN_SCALE_BRAND_PRODUCT)
            
            if cls_name == "barcode":
                barcode_value = decode_barcode_image(cropped)
                if not barcode_value or not barcode_value.strip():
                    print("Barcode data is unreadable; using OCR as fallback for barcode region.")
                    barcode_value = ocr_read_text(cropped, ocr_reader, scale_factor=DEFAULT_SCALE_FACTOR)
                    if not barcode_value or not barcode_value.strip():
                        print("Fallback OCR also failed for barcode region.")
                        barcode_value = "Barcode unreadable"
                output_data["barcode"].append(barcode_value)
            else:
                text = ocr_read_text(cropped, ocr_reader, scale_factor=scale)
                output_data[cls_name].append(text)
    else:
        print("No detections found by YOLO.")
    
    # Run full-image OCR as backup.
    print("Running full-image OCR for additional text detection.")
    full_text = ocr_read_text(original_img, ocr_reader, scale_factor=1.0)
    output_data["all_text"] = full_text

    return output_data

if __name__ == "__main__":
    # Teting path
    # Run17 best one
    # Run19 the best one so far, major improvements (still need a larger daatset to classify better tho)
    model_path = "/Users/allenpereira/Desktop/image_git/runs/detect/new_run19"
    test_image_path = "/Users/allenpereira/Desktop/image_git/WhatsApp Image 2025-03-16 at 7.13.31 PM.jpeg"
    
    model = load_model(model_path)
    results = predict_on_image(model, test_image_path)
    print("Final Output:")
    print(results)



pip install -r requirements.txt

▶Run the API

Start the FastAPI server locally:

uvicorn app:app --reload --host 0.0.0.0 --port 8000

Testing the Endpoints

1. Predict (File Upload)

Use curl:

curl -X POST "http://localhost:8000/predict/" \
     -F "file=@/path/to/your/image.jpg" \
     -F "conf=0.6"

Or in React/TypeScript with fetch:

const form = new FormData();
form.append('file', imageFile);
form.append('conf', '0.6');

const res = await fetch('http://localhost:8000/predict/', {
  method: 'POST',
  body: form,
});
const data = await res.json();
console.log(data);

2. Predict (URL-based JSON)

Serve your image locally or use a public URL:

curl -X POST "http://localhost:8000/predict_url/" \
     -H "Content-Type: application/json" \
     -d '{ "url": "http://localhost:8001/your-image.jpg", "conf": 0.5 }'

Or in React/TypeScript:

const payload = { url: imageUrl, conf: 0.5 };

const res = await fetch('http://localhost:8000/predict_url/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
const data = await res.json();
console.log(data);

3. Train YOLO Model

Trigger a training run (default: 50 epochs, batch size 10, learning rate 0.01):

curl -X POST "http://localhost:8000/train/?epochs=20&batch=8&lr0=0.001"

📁 Project Structure

train.py — Script to train or fine-tune the YOLO model on your dataset.

predict.py — Contains predict_on_image() for running inference on single images.

app.py — FastAPI application exposing /predict/, /predict_url/, and /train/ endpoints.

packaging_data.yaml — Configuration for dataset locations and class names.

requirements.txt — Python dependencies for easy setup.

📐 API Endpoints

POST /predict/
http://localhost:8000/predict/


Parameters (multipart/form-data):

file: image file upload (JPEG, PNG, WebP supported)

conf (optional): confidence threshold (default: 0.5)

Response: JSON with arrays for barcode, brand, price, description, product_name, and all_text.

POST /predict_url/

Parameters (application/json):

url: publicly accessible or locally served image URL

conf (optional): confidence threshold (default: 0.5)

Response: same JSON schema as /predict/.

POST /train/

Parameters (query):

epochs: number of training epochs (default: 50)

batch: batch size (default: 10)

lr0: initial learning rate (default: 0.01)

Response: status message confirming training start.




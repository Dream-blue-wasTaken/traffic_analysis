import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="VisionAnalytica YOLO Backend")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model from Hugging Face
MODEL_URL = "https://huggingface.co/nnsohamnn/helmet-detection-yolo11/resolve/main/yolov11m%28100epochs%29.pt"
print(f"📥 Loading model from {MODEL_URL}...")
try:
    model = YOLO(MODEL_URL)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    # Fallback to local if download fails or is already there
    model = None

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        # Convert RGB to BGR for OpenCV/Ultralytics if needed (ultralytics usually handles PIL/numpy RGB)
        
        # Run inference
        results = model(img_np, conf=0.35)[0]
        
        detections = []
        for box in results.boxes:
            # box.xyxy is [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            detections.append({
                "box": coords,
                "confidence": conf,
                "label": label,
                "class_id": cls_id
            })
            
        return {
            "detections": detections,
            "image_size": {
                "width": img_np.shape[1],
                "height": img_np.shape[0]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

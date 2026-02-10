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

# Load Models
MODEL_URL = "https://huggingface.co/nnsohamnn/helmet-detection-yolo11/resolve/main/yolov11m%28100epochs%29.pt"
print(f"📥 Loading helmet model from {MODEL_URL}...")
try:
    helmet_model = YOLO(MODEL_URL)
    # Load a faster model for general person/vehicle detection (COCO classes)
    base_model = YOLO("yolo11n.pt") 
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    helmet_model = None

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    if not helmet_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        h_orig, w_orig = img_np.shape[:2]

        # --- STAGE 1: Detect Persons and Motorcycles (COCO classes: 0=person, 3=motorcycle) ---
        base_results = base_model(img_np, conf=0.3, imgsz=640)[0]
        
        persons = []
        motorcycles = []
        
        for box in base_results.boxes:
            cls = int(box.cls[0])
            if cls == 0: persons.append(box.xyxy[0].tolist())
            elif cls == 3: motorcycles.append(box.xyxy[0].tolist())

        # --- STAGE 2: Rider Matching & Cropping ---
        # We'll focus on people who are likely riders or near bikes
        all_detections = []
        
        for p_box in persons:
            px1, py1, px2, py2 = p_box
            p_w = px2 - px1
            p_h = py2 - py1
            
            # Crop the "Head/Shoulder" area (top 40% of the person box)
            # We add a bit of padding to ensure the whole helmet fits
            cx1 = max(0, int(px1 - p_w * 0.1))
            cy1 = max(0, int(py1 - p_h * 0.1))
            cx2 = min(w_orig, int(px2 + p_w * 0.1))
            cy2 = min(h_orig, int(py1 + p_h * 0.5)) # Crop upper half
            
            crop = img_np[cy1:cy2, cx1:cx2]
            if crop.size == 0: continue

            # --- STAGE 3: Specialized Helmet Detection on Crop ---
            h_results = helmet_model(crop, conf=0.2, imgsz=320)[0]
            
            for h_box in h_results.boxes:
                hx1, hy1, hx2, hy2 = h_box.xyxy[0].tolist()
                conf = float(h_box.conf[0])
                cls_id = int(h_box.cls[0])
                label = helmet_model.names[cls_id]

                # Map back to original coordinates
                final_box = [
                    hx1 + cx1,
                    hy1 + cy1,
                    hx2 + cx1,
                    hy2 + cy1
                ]

                all_detections.append({
                    "box": final_box,
                    "confidence": conf,
                    "label": label,
                    "class_id": cls_id
                })

        return {
            "detections": apply_nms(all_detections, iou_threshold=0.45),
            "image_size": {
                "width": w_orig,
                "height": h_orig
            }
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def apply_nms(detections, iou_threshold=0.45):
    """
    Simple Non-Maximum Suppression to remove overlapping boxes.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # Check IOU with remaining boxes
        remaining = []
        for det in sorted_dets:
            if calculate_iou(best['box'], det['box']) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
        
    return keep

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IOU) of two boxes.
    Each box is [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0: return 0
    
    return intersection / union

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

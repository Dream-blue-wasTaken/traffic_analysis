# VisionAnalytica 
### AI-Powered Industrial Safety & Traffic Analysis

VisionAnalytica is a state-of-the-art computer vision application designed to enhance safety in industrial and urban environments. It leverages the power of **YOLOv11** to detect individuals, motorcycles, and verify if riders are wearing safety helmets.

---

###  Tech Stack

- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS (CDN)
- **Backend**: FastAPI, Python 3.10+, Ultralytics (YOLOv11), Pillow
- **Deep Learning**: 
  - Base: `yolo11n.pt`
  - Specialized: `helmet-detection-yolo11` (from HuggingFace)

---

###  Getting Started

#### 1. Clone the Repository
```bash
git clone https://github.com/Dream-blue-wasTaken/traffic_analysis.git
cd visionanalytica
```

#### 2. Backend Setup
The backend handles the AI inference.

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r ../requirements.txt

# Run the backend server
python main.py
```
*The server will start at `http://localhost:8000`.*

#### 3. Frontend Setup
The frontend provides the user interface.

```bash
# Navigate to frontend (from root)
cd frontend

# Install dependencies
npm install

# Run the app
npm run dev
```
*Access the application at `http://localhost:5173` (or the port Vite provides).*

---

### 🔍 How it Works

1. **User Uploads Image**: The frontend sends the image to the FastAPI `/detect` endpoint.
2. **Base Detection**: The backend first identifies all `person` and `motorcycle` instances using YOLOv11n.
3. **Region of Interest (ROI) Extraction**: For every person detected, the system extracts the head and upper shoulder region (using intelligent padding).
4. **Specialized Classification**: The cropped ROI is processed by a specialized model to determine `helmet` vs. `no-helmet`.
5. **Coordinate Mapping**: Detections are mapped back to the original image coordinates and displayed on the UI.

---

### 📂 Repository Structure

- `backend/main.py`: Core API logic and two-stage detection pipeline.
- `frontend/`: React application using Tailwind CSS for a sleek, modern UI.
- `requirements.txt`: Python package dependencies.
- `weights/`: Directory where downloaded models are cached.

### 🛠️ Prerequisites
- **Node.js**: v18+ 
- **Python**: 3.9+
- **GPU (Optional)**: For faster inference (requires CUDA-enabled PyTorch)

---
import os
import cv2
import json
import base64
import subprocess
import numpy as np
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .vehicle_detector import VehicleDetector
from .plate_detector import PlateDetector

app = FastAPI(
    title="Vehicle and Number Plate Detection API",
    description="Two-stage vehicle and license plate detection system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once when API starts
vehicle_detector = VehicleDetector()
plate_detector = PlateDetector()

PADDLE_PYTHON = (
    r"W:\Tfrenzy\Vehicle Detection"
    r"\paddle_env\Scripts\python.exe"
)
OCR_WORKER = "ocr_worker.py"

def run_ocr(image_path):
    result = subprocess.run(
        [
            PADDLE_PYTHON,
            OCR_WORKER,
            image_path
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {"text": "", "confidence": 0.0}

    try:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "text" in data:
                    return data
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return {"text": "", "confidence": 0.0}


@app.get("/")
def root():
    return {
        "message": "Vehicle and Number Plate Detection API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "vehicle_detector": "ready",
        "plate_detector": "ready",
        "ocr": "ready"
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):
    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    # We will draw on this copy to return an annotated image
    annotated_image = image.copy()

    vehicles = vehicle_detector.detect(image)
    response = []

    # Temporary directory for plate crops to pass to ocr_worker
    temp_dir = tempfile.mkdtemp()

    for i, vehicle in enumerate(vehicles):
        x1, y1, x2, y2 = vehicle["bbox"]
        
        # Absolute vehicle coords
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        vehicle_crop = image[y1:y2, x1:x2]
        
        # Draw vehicle box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        v_label = f"{vehicle['type']} {vehicle['confidence']:.2f}"
        cv2.putText(annotated_image, v_label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        vehicle_result = {
            "vehicle_type": vehicle["type"],
            "vehicle_confidence": vehicle["confidence"],
            "vehicle_bbox": [x1, y1, x2, y2],
            "plate": None
        }

        if vehicle_crop.size > 0:
            plates = plate_detector.detect(vehicle_crop)
            if plates:
                plate = max(plates, key=lambda x: x["confidence"])
                px1, py1, px2, py2 = plate["bbox"]

                # Expand plate crop slightly
                plate_width = px2 - px1
                plate_height = py2 - py1
                padding_x = int(plate_width * 0.15)
                padding_y = int(plate_height * 0.25)

                px1 = max(0, px1 - padding_x)
                py1 = max(0, py1 - padding_y)
                px2 = min(vehicle_crop.shape[1], px2 + padding_x)
                py2 = min(vehicle_crop.shape[0], py2 + padding_y)

                plate_crop = vehicle_crop[py1:py2, px1:px2]

                if plate_crop.size > 0:
                    plate_path = os.path.join(temp_dir, f"plate_{i}.jpg")
                    cv2.imwrite(plate_path, plate_crop)
                    
                    ocr_result = run_ocr(plate_path)
                    text = ocr_result.get("text", "")
                    ocr_conf = float(ocr_result.get("confidence", 0.0))

                    abs_px1 = x1 + px1
                    abs_py1 = y1 + py1
                    abs_px2 = x1 + px2
                    abs_py2 = y1 + py2

                    # Draw plate box
                    cv2.rectangle(annotated_image, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 0, 255), 2)
                    
                    p_label = f"{text} {ocr_conf:.2f}" if text else f"plate {plate['confidence']:.2f}"
                    cv2.putText(annotated_image, p_label, (abs_px1, max(20, abs_py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    vehicle_result["plate"] = {
                        "confidence": plate["confidence"],
                        "bbox": [abs_px1, abs_py1, abs_px2, abs_py2],
                        "text": text,
                        "ocr_confidence": ocr_conf
                    }

        response.append(vehicle_result)

    # Encode annotated image to base64 for frontend
    _, buffer = cv2.imencode('.jpg', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    base64_img_str = f"data:image/jpeg;base64,{encoded_image}"

    return {
        "success": True,
        "vehicle_count": len(response),
        "vehicles": response,
        "annotated_image": base64_img_str
    }
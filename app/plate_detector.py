import os

from ultralytics import YOLO

from .config import DEVICE, PLATE_CONFIDENCE


class PlateDetector:

    def __init__(self, model_path="models/plate.pt"):

        self.model = None

        if not os.path.exists(model_path):

            print(
                f"[WARNING] Plate model not found: {model_path}"
            )

            print(
                "[WARNING] Plate detection is currently disabled."
            )

            return

        print("[INFO] Loading plate detection model...")

        self.model = YOLO(model_path)

        print("[INFO] Plate model loaded.")

    def detect(self, vehicle_image):

        if self.model is None:
            return []

        results = self.model.predict(
            source=vehicle_image,
            conf=PLATE_CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        plates = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                plates.append({
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                })

        return plates
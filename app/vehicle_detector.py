from ultralytics import YOLO

from .config import DEVICE, VEHICLE_CONFIDENCE


class VehicleDetector:

    def __init__(self, model_path="yolo26n.pt"):

        print("[INFO] Loading vehicle detection model...")

        self.model = YOLO(model_path)

        # COCO vehicle classes
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        print("[INFO] Vehicle model loaded.")

    def detect(self, image):

        results = self.model.predict(
            source=image,
            conf=VEHICLE_CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        vehicles = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Ignore non-vehicle objects
                if class_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                vehicles.append({
                    "type": self.vehicle_classes[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                })

        return vehicles
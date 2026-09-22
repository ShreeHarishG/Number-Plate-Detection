import os
import cv2
import json
import subprocess

from app.vehicle_detector import VehicleDetector
from app.plate_detector import PlateDetector


IMAGE_PATH = "test/image.png"
OUTPUT_PATH = "outputs/final_detection.jpg"
PLATE_OUTPUT_DIR = "outputs/plates"

PADDLE_PYTHON = (
    r"W:\Tfrenzy\Vehicle Detection"
    r"\paddle_env\Scripts\python.exe"
)

OCR_WORKER = "ocr_worker.py"


os.makedirs("outputs", exist_ok=True)
os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)


def run_ocr(image_path):
    """
    Run PaddleOCR in the separate paddle_env.
    """

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
        print("[ERROR] PaddleOCR failed:")
        print(result.stderr)

        return {
            "text": "",
            "confidence": 0.0
        }

    try:
        # PaddleOCR may print startup logs.
        # Find the final JSON line.
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        for line in reversed(lines):
            try:
                data = json.loads(line)

                if isinstance(data, dict) and "text" in data:
                    return data

            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"[ERROR] Could not parse OCR result: {e}")

    return {
        "text": "",
        "confidence": 0.0
    }


print()
print("[INFO] Loading image...")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not find image: {IMAGE_PATH}"
    )

print(
    f"[INFO] Image loaded: "
    f"{image.shape[1]}x{image.shape[0]}"
)


print()
print("[INFO] Loading vehicle detector...")

vehicle_detector = VehicleDetector()


print()
print("[INFO] Loading plate detector...")

plate_detector = PlateDetector(
    model_path="models/plate.pt"
)


print()
print("=" * 70)
print("STAGE 1 + STAGE 2: VEHICLE + PLATE DETECTION")
print("=" * 70)


vehicles = vehicle_detector.detect(image)

print(
    f"Vehicles detected: {len(vehicles)}"
)


total_plates = 0
successful_ocr = 0


for vehicle_index, vehicle in enumerate(
    vehicles,
    start=1
):

    x1, y1, x2, y2 = vehicle["bbox"]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    vehicle_crop = image[
        y1:y2,
        x1:x2
    ]

    if vehicle_crop.size == 0:
        print(
            f"Vehicle {vehicle_index}: "
            f"Invalid vehicle crop"
        )
        continue


    # -------------------------------------------------
    # PLATE DETECTION
    # -------------------------------------------------

    plates = plate_detector.detect(
        vehicle_crop
    )


    if not plates:

        print(
            f"Vehicle {vehicle_index}: "
            f"{vehicle['type']} "
            f"| vehicle={vehicle['confidence']:.2f} "
            f"| plate=NOT FOUND"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"{vehicle['type']} - no plate",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        continue


    # Keep highest-confidence plate

    plate = max(
        plates,
        key=lambda x: x["confidence"]
    )

    total_plates += 1


    px1, py1, px2, py2 = plate["bbox"]

    px1 = max(
        0,
        px1
    )

    py1 = max(
        0,
        py1
    )

    px2 = min(
        vehicle_crop.shape[1],
        px2
    )

    py2 = min(
        vehicle_crop.shape[0],
        py2
    )


    # -------------------------------------------------
    # EXPAND PLATE CROP
    # -------------------------------------------------

    plate_width = px2 - px1
    plate_height = py2 - py1

    padding_x = int(
        plate_width * 0.15
    )

    padding_y = int(
        plate_height * 0.25
    )

    px1 = max(
        0,
        px1 - padding_x
    )

    py1 = max(
        0,
        py1 - padding_y
    )

    px2 = min(
        vehicle_crop.shape[1],
        px2 + padding_x
    )

    py2 = min(
        vehicle_crop.shape[0],
        py2 + padding_y
    )


    plate_crop = vehicle_crop[
        py1:py2,
        px1:px2
    ]


    if plate_crop.size == 0:

        print(
            f"Vehicle {vehicle_index}: "
            f"Invalid plate crop"
        )

        continue


    # -------------------------------------------------
    # SAVE PLATE
    # -------------------------------------------------

    plate_path = os.path.join(
        PLATE_OUTPUT_DIR,
        f"vehicle_{vehicle_index}.jpg"
    )

    cv2.imwrite(
        plate_path,
        plate_crop
    )


    # -------------------------------------------------
    # OCR
    # -------------------------------------------------

    print(
        f"[INFO] Running PaddleOCR "
        f"for vehicle {vehicle_index}..."
    )

    ocr_result = run_ocr(
        plate_path
    )

    text = ocr_result.get(
        "text",
        ""
    )

    ocr_confidence = float(
        ocr_result.get(
            "confidence",
            0.0
        )
    )


    if text:
        successful_ocr += 1


    # -------------------------------------------------
    # ABSOLUTE PLATE COORDINATES
    # -------------------------------------------------

    absolute_x1 = x1 + px1
    absolute_y1 = y1 + py1

    absolute_x2 = x1 + px2
    absolute_y2 = y1 + py2


    print(
        f"Vehicle {vehicle_index}: "
        f"{vehicle['type']} "
        f"| vehicle={vehicle['confidence']:.2f} "
        f"| plate={plate['confidence']:.2f} "
        f"| OCR={ocr_confidence:.2f} "
        f"| TEXT={text if text else 'NOT READ'}"
    )


    # -------------------------------------------------
    # DRAW VEHICLE
    # -------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    vehicle_label = (
        f"{vehicle['type']} "
        f"{vehicle['confidence']:.2f}"
    )

    cv2.putText(
        image,
        vehicle_label,
        (
            x1,
            max(20, y1 - 10)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # -------------------------------------------------
    # DRAW PLATE
    # -------------------------------------------------

    cv2.rectangle(
        image,
        (
            absolute_x1,
            absolute_y1
        ),
        (
            absolute_x2,
            absolute_y2
        ),
        (0, 0, 255),
        2
    )


    display_text = (
        f"{text} {ocr_confidence:.2f}"
        if text
        else f"plate {plate['confidence']:.2f}"
    )


    cv2.putText(
        image,
        display_text,
        (
            absolute_x1,
            max(
                20,
                absolute_y1 - 5
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        2
    )


# -----------------------------------------------------
# SAVE FINAL IMAGE
# -----------------------------------------------------

cv2.imwrite(
    OUTPUT_PATH,
    image
)


print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Vehicles detected : {len(vehicles)}"
)

print(
    f"Plates detected   : {total_plates}"
)

print(
    f"OCR successful    : {successful_ocr}"
)

print(
    f"Output image      : {OUTPUT_PATH}"
)

print(
    f"Plate crops       : {PLATE_OUTPUT_DIR}"
)

print("=" * 70)
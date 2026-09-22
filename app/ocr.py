import torch
import cv2
from paddleocr import PaddleOCR


class PlateOCR:
    def __init__(self):
        print("[INFO] Loading PaddleOCR model...")

        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0"
        )

        print("[INFO] PaddleOCR model loaded.")

    def preprocess(self, image):
        # Enlarge small license plate
        image = cv2.resize(
            image,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Improve contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        return gray

    def read(self, image):

        if image is None or image.size == 0:
            return {
                "text": "",
                "confidence": 0.0
            }

        processed = self.preprocess(image)

        results = self.ocr.predict(processed)

        best_text = ""
        best_confidence = 0.0

        for result in results:

            try:
                data = result.json
            except Exception:
                data = result

            if isinstance(data, dict):

                if "res" in data:
                    data = data["res"]

                texts = data.get(
                    "rec_texts",
                    []
                )

                scores = data.get(
                    "rec_scores",
                    []
                )

                for text, score in zip(
                    texts,
                    scores
                ):

                    confidence = float(score)

                    if confidence > best_confidence:
                        best_text = str(text)
                        best_confidence = confidence

        # Normalize
        best_text = best_text.upper()

        best_text = "".join(
            c
            for c in best_text
            if c.isalnum()
        )

        return {
            "text": best_text,
            "confidence": round(
                best_confidence,
                4
            )
        }
import sys
import json
import cv2

from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu:0"
)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "text": "",
            "confidence": 0.0,
            "error": "No image path provided"
        }))
        return

    image_path = sys.argv[1]

    image = cv2.imread(image_path)

    if image is None:
        print(json.dumps({
            "text": "",
            "confidence": 0.0,
            "error": "Could not read image"
        }))
        return

    image = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    processed = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    results = ocr.predict(processed)

    all_texts = []
    total_score = 0.0
    valid_count = 0

    for result in results:
        try:
            data = result.json
        except Exception:
            data = result

        if isinstance(data, dict):
            if "res" in data:
                data = data["res"]

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

            for text, score in zip(texts, scores):
                score = float(score)
                if score > 0.3:  # Only keep somewhat confident text
                    all_texts.append(str(text))
                    total_score += score
                    valid_count += 1

    if valid_count > 0:
        raw_text = " ".join(all_texts)
        best_confidence = total_score / valid_count
    else:
        raw_text = ""
        best_confidence = 0.0

    # Clean text: uppercase, keep alphanumeric and spaces
    clean_text = raw_text.upper()
    clean_text = "".join(c for c in clean_text if c.isalnum() or c.isspace())
    
    # Remove extra spaces
    best_text = " ".join(clean_text.split())

    print(json.dumps({
        "text": best_text,
        "confidence": round(best_confidence, 4)
    }))


if __name__ == "__main__":
    main()
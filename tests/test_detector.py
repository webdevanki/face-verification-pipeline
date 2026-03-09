"""
Test detektora twarzy - uruchom z głównego folderu projektu:
    python tests/test_detector.py
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.detector import FaceDetector


def test_detector():
    img_path = "data/raw/test.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print(f"BŁĄD: Nie znaleziono pliku {img_path}")
        print("Wrzuć dowolne zdjęcie z twarzą do data/raw/test.jpg")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Wczytano obraz: {img.shape[1]}x{img.shape[0]}px")

    detector = FaceDetector()

    print("\n--- Wszystkie twarze ---")
    faces = detector.detect(img)
    print(f"Wykryto twarzy: {len(faces)}")
    for i, face in enumerate(faces):
        print(f"  Twarz {i+1}: confidence={face.confidence:.2f}, bbox={face.bbox}, crop={face.crop.shape}")

    print("\n--- Największa twarz ---")
    face = detector.detect_largest(img)
    if face:
        print(f"OK  confidence={face.confidence:.2f}, bbox={face.bbox}, crop shape={face.crop.shape}")

        # zapisz wyciętą twarz żeby zobaczyć wynik
        output_path = "data/processed/detected_face.jpg"
        Path("data/processed").mkdir(exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(face.crop, cv2.COLOR_RGB2BGR))
        print(f"OK  wycieta twarz zapisana: {output_path}")
    else:
        print("BŁĄD: Brak wykrytej twarzy")


if __name__ == "__main__":
    print("=== Test detektora twarzy ===\n")
    test_detector()

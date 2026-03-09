"""
Test embeddingów twarzy - uruchom z głównego folderu projektu:
    python tests/test_embedder.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.detector import FaceDetector
from src.pipeline.embedder import FaceEmbedder, cosine_similarity, euclidean_distance


def get_face_crop(img_path: str, detector: FaceDetector) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = detector.detect_largest(img)
    if face is None:
        raise ValueError(f"Brak twarzy w {img_path}")
    return face.crop


def test_same_image():
    """To samo zdjęcie = identyczne embeddingi = similarity ~1.0"""
    detector = FaceDetector()
    embedder = FaceEmbedder()

    crop = get_face_crop("data/raw/test.jpg", detector)
    emb1 = embedder.extract(crop)
    emb2 = embedder.extract(crop)

    sim = cosine_similarity(emb1, emb2)
    dist = euclidean_distance(emb1, emb2)

    print(f"To samo zdjęcie:")
    print(f"  cosine similarity = {sim:.4f}  (oczekiwane: ~1.0)")
    print(f"  euclidean distance = {dist:.4f}  (oczekiwane: ~0.0)")
    assert sim > 0.99, f"Powinno być ~1.0, got {sim}"
    print("OK")


def test_embedding_shape():
    """Embedding powinien mieć 512 wymiarów i być znormalizowany."""
    detector = FaceDetector()
    embedder = FaceEmbedder()

    crop = get_face_crop("data/raw/test.jpg", detector)
    emb = embedder.extract(crop)

    print(f"\nKształt embeddingu: {emb.vector.shape}")
    print(f"Norma wektora: {np.linalg.norm(emb.vector):.4f}  (oczekiwane: ~1.0)")

    assert emb.vector.shape[0] == 512, f"Oczekiwano 512 wymiarów, got {emb.vector.shape[0]}"
    assert abs(np.linalg.norm(emb.vector) - 1.0) < 1e-5, "Wektor nie jest znormalizowany"
    print("OK")


def test_different_images():
    """
    Dwa różne zdjęcia tej samej osoby.
    Potrzebujesz data/raw/test2.jpg - drugie zdjęcie tej samej osoby.
    """
    test2_path = "data/raw/test2.jpg"
    if not Path(test2_path).exists():
        print(f"\nPomiń test różnych zdjęć - brak {test2_path}")
        print("Wrzuć drugie zdjęcie tej samej osoby jako data/raw/test2.jpg")
        return

    detector = FaceDetector()
    embedder = FaceEmbedder()

    crop1 = get_face_crop("data/raw/test.jpg", detector)
    crop2 = get_face_crop(test2_path, detector)

    emb1 = embedder.extract(crop1)
    emb2 = embedder.extract(crop2)

    sim = cosine_similarity(emb1, emb2)
    print(f"\nDwa zdjęcia tej samej osoby:")
    print(f"  cosine similarity = {sim:.4f}  (oczekiwane: >0.4 = ta sama osoba)")
    if sim > 0.4:
        print("OK  model rozpoznał tę samą osobę")
    else:
        print(f"UWAGA: niska similarity ({sim:.4f}) - może różne osoby lub słaba jakość zdjęcia")


if __name__ == "__main__":
    print("=== Test embeddingów twarzy ===\n")
    test_embedding_shape()
    test_same_image()
    test_different_images()
    print("\nGotowe!")

"""
Test preprocessora - uruchom z głównego folderu projektu:
    python tests/test_preprocessor.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.preprocessor import ImagePreprocessor


def make_fake_image(brightness: int = 128, blur: bool = False) -> np.ndarray:
    """Tworzy syntetyczny obraz do testów - bez potrzeby prawdziwych zdjęć."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * brightness
    if not blur:
        # dodaj krawędzie żeby obraz był "ostry"
        img[50:150, 50:150] = brightness + 40
        img[80:120, 80:120] = brightness - 40
    return img


def test_quality_check():
    preprocessor = ImagePreprocessor()

    # obraz dobrej jakości
    good_img = make_fake_image(brightness=128, blur=False)
    report = preprocessor.check_quality(good_img, "test_good.jpg")
    assert report.is_valid, f"Powinien przejść: {report.reject_reason}"
    print(f"OK  dobry obraz: brightness={report.brightness:.1f}, sharpness={report.blurriness:.1f}")

    # zbyt ciemny obraz
    dark_img = make_fake_image(brightness=10)
    report = preprocessor.check_quality(dark_img, "test_dark.jpg")
    assert not report.is_valid
    print(f"OK  ciemny obraz odrzucony: {report.reject_reason}")

    # za mały obraz
    tiny_img = np.zeros((30, 30, 3), dtype=np.uint8)
    report = preprocessor.check_quality(tiny_img, "test_tiny.jpg")
    assert not report.is_valid
    print(f"OK  mały obraz odrzucony: {report.reject_reason}")


def test_normalize():
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    img = make_fake_image()

    normalized = preprocessor.normalize(img)

    assert normalized.shape == (640, 640, 3), f"Zły kształt: {normalized.shape}"
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    print(f"OK  normalizacja: shape={normalized.shape}, zakres=[{normalized.min():.2f}, {normalized.max():.2f}]")


if __name__ == "__main__":
    print("=== Test preprocessora ===\n")
    test_quality_check()
    test_normalize()
    print("\nWszystkie testy przeszły!")

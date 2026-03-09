"""
Face detection pipeline.

Wykrywa twarze na obrazie i zwraca przycięte zdjęcia twarzy
gotowe do ekstrakcji embeddingów.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import cv2


@dataclass
class DetectedFace:
    """Pojedyncza wykryta twarz."""
    bbox: np.ndarray        # [x1, y1, x2, y2]
    confidence: float       # pewność detekcji 0-1
    crop: np.ndarray        # przycięty obraz twarzy (112x112)
    keypoints: Optional[np.ndarray] = None  # 5 punktów: oczy, nos, kąciki ust


class FaceDetector:
    """
    Detektor twarzy oparty na InsightFace (model RetinaFace).

    RetinaFace to jeden z najdokładniejszych detektorów twarzy.
    Używany w produkcyjnych systemach biometrycznych.
    """

    def __init__(self, min_confidence: float = 0.5, target_size: int = 112):
        """
        Args:
            min_confidence: minimalny próg pewności detekcji (0-1)
            target_size: rozmiar wyciętej twarzy w pikselach (standard biometryczny: 112x112)
        """
        self.min_confidence = min_confidence
        self.target_size = target_size
        self._app = None

    def _load_model(self):
        """Lazy loading modelu - ładuje się tylko przy pierwszym użyciu."""
        if self._app is not None:
            return

        import insightface
        from insightface.app import FaceAnalysis

        print("Ładowanie modelu detekcji (pierwsze uruchomienie pobierze ~200MB)...")
        self._app = FaceAnalysis(
            name="buffalo_sc",      # lekki model CPU-friendly
            providers=["CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        print("Model załadowany.")

    def detect(self, img: np.ndarray) -> list[DetectedFace]:
        """
        Wykrywa wszystkie twarze na obrazie.

        Args:
            img: obraz RGB jako numpy array

        Returns:
            lista wykrytych twarzy, posortowana po confidence (najlepsza pierwsza)
        """
        self._load_model()

        # InsightFace wymaga BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = self._app.get(img_bgr)

        results = []
        for face in faces:
            if face.det_score < self.min_confidence:
                continue

            bbox = face.bbox.astype(int)
            crop = self._crop_face(img, bbox)
            if crop is None:
                continue

            results.append(DetectedFace(
                bbox=bbox,
                confidence=float(face.det_score),
                crop=crop,
                keypoints=face.kps if hasattr(face, 'kps') else None,
            ))

        # sortuj po confidence - najlepsza twarz pierwsza
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def detect_largest(self, img: np.ndarray) -> Optional[DetectedFace]:
        """
        Zwraca największą twarz na obrazie.

        W systemach biometrycznych zakładamy że osoba weryfikowana
        jest najbliżej kamery = ma największą twarz.
        """
        faces = self.detect(img)
        if not faces:
            return None

        # największa twarz = największe pole bounding boxa
        return max(faces, key=lambda f: self._bbox_area(f.bbox))

    def _crop_face(self, img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Wycina i skaluje twarz do target_size x target_size."""
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]

        # upewnij się że bbox jest w granicach obrazu
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop, (self.target_size, self.target_size))

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> int:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

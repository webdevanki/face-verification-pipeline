"""
Face embedding extraction pipeline.

Zamienia obraz twarzy (112x112) na wektor liczbowy (embedding).
Embeddingi służą do porównywania twarzy bez przechowywania zdjęć.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class FaceEmbedding:
    """Embedding pojedynczej twarzy."""
    vector: np.ndarray      # wektor 512 liczb float32
    norm: float             # długość wektora (powinna być ~1.0 po normalizacji)
    source_path: str = ""   # skąd pochodzi zdjęcie


class FaceEmbedder:
    """
    Ekstraktor embeddingów twarzy oparty na modelu ArcFace.

    ArcFace to state-of-the-art model do rozpoznawania twarzy.
    Trenowany na milionach twarzy, zwraca 512-wymiarowy wektor.

    Dlaczego 512 wymiarów? To kompromis między dokładnością a szybkością.
    Każdy wymiar koduje jakąś cechę twarzy (odległość oczu, kształt nosa itp.)
    - ale nie dosłownie, model sam decyduje co kodować podczas treningu.
    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Lazy loading - model ładuje się tylko przy pierwszym użyciu."""
        if self._model is not None:
            return

        import insightface
        from insightface.app import FaceAnalysis

        print("Ładowanie modelu ArcFace...")
        app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))

        # wyciągamy sam model rozpoznawania (rec = recognition)
        self._model = app.models.get("w600k_mbf")
        if self._model is None:
            # fallback - weź pierwszy dostępny model rec
            for name, model in app.models.items():
                if hasattr(model, 'get_feat'):
                    self._model = model
                    break

        print("Model ArcFace załadowany.")

    def extract(self, face_crop: np.ndarray) -> Optional[FaceEmbedding]:
        """
        Ekstrahuje embedding z wyciętej twarzy.

        Args:
            face_crop: obraz twarzy RGB 112x112

        Returns:
            FaceEmbedding z znormalizowanym wektorem, lub None przy błędzie
        """
        self._load_model()

        if face_crop.shape[:2] != (112, 112):
            import cv2
            face_crop = cv2.resize(face_crop, (112, 112))

        try:
            # model oczekuje float32 w zakresie [0, 255] - nie normalizujemy tutaj
            if face_crop.dtype != np.uint8:
                face_crop = (face_crop * 255).astype(np.uint8)

            vector = self._model.get_feat(face_crop)
            vector = vector.flatten()

            # L2 normalizacja - wektor jednostkowy
            # Po normalizacji cosine similarity == dot product - szybsze obliczenia
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            return FaceEmbedding(vector=vector, norm=float(norm))

        except Exception as e:
            print(f"Błąd ekstrakcji embeddingu: {e}")
            return None

    def extract_batch(self, face_crops: list[np.ndarray]) -> list[Optional[FaceEmbedding]]:
        """Przetwarza listę twarzy."""
        return [self.extract(crop) for crop in face_crops]


def cosine_similarity(emb1: FaceEmbedding, emb2: FaceEmbedding) -> float:
    """
    Podobieństwo cosinusowe między dwoma embeddingami.

    Zakres: -1 (przeciwne) do 1 (identyczne).
    W praktyce dla twarzy: >0.4 = ta sama osoba, <0.4 = różne osoby.

    Po normalizacji L2 to po prostu dot product - dlatego normalizujemy.
    """
    return float(np.dot(emb1.vector, emb2.vector))


def euclidean_distance(emb1: FaceEmbedding, emb2: FaceEmbedding) -> float:
    """
    Odległość euklidesowa między embeddingami.

    Im mniejsza, tym bardziej podobne twarze.
    Alternatywa dla cosine similarity.
    """
    return float(np.linalg.norm(emb1.vector - emb2.vector))

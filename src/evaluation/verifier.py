"""
End-to-end face verification pipeline.

Łączy wszystkie komponenty w jeden spójny system:
preprocessor → detector → embedder → decision
"""

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.pipeline.preprocessor import ImagePreprocessor
from src.pipeline.detector import FaceDetector
from src.pipeline.embedder import FaceEmbedder, FaceEmbedding, cosine_similarity
from src.metrics.biometric import compute_biometric_metrics, print_metrics_report, BiometricMetrics


@dataclass
class VerificationResult:
    """Wynik weryfikacji pary zdjęć."""
    img1_path: str
    img2_path: str
    similarity: float
    decision: bool              # True = ta sama osoba
    threshold: float
    error: Optional[str] = None  # jeśli coś poszło nie tak


class FaceVerifier:
    """
    System weryfikacji twarzy end-to-end.

    Weryfikacja vs identyfikacja:
    - Weryfikacja: "Czy to jest ta sama osoba co na referencyjnym zdjęciu?" (1:1)
    - Identyfikacja: "Kto to jest spośród N osób w bazie?" (1:N)

    Ten system implementuje weryfikację (prostszy i bezpieczniejszy przypadek).
    """

    def __init__(self, threshold: float = 0.45):
        """
        Args:
            threshold: próg decyzyjny similarity score.
                       Powyżej → ta sama osoba, poniżej → różne osoby.
                       Domyślnie 0.45 - typowy kompromis FAR/FRR dla ArcFace.
        """
        self.threshold = threshold
        self.preprocessor = ImagePreprocessor()
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

    def _load_and_embed(self, img_path: str) -> tuple[Optional[FaceEmbedding], Optional[str]]:
        """
        Wczytuje obraz, wykrywa twarz i ekstrahuje embedding.

        Returns:
            (embedding, error_message) - error_message jest None jeśli sukces
        """
        img = cv2.imread(img_path)
        if img is None:
            return None, f"Nie można wczytać: {img_path}"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # quality check
        report = self.preprocessor.check_quality(img, img_path)
        if not report.is_valid:
            return None, f"Zła jakość obrazu: {report.reject_reason}"

        # detekcja twarzy
        face = self.detector.detect_largest(img)
        if face is None:
            return None, f"Brak twarzy: {img_path}"

        # embedding
        embedding = self.embedder.extract(face.crop)
        if embedding is None:
            return None, f"Błąd ekstrakcji embeddingu: {img_path}"

        embedding.source_path = img_path
        return embedding, None

    def verify(self, img1_path: str, img2_path: str) -> VerificationResult:
        """
        Weryfikuje czy dwa zdjęcia przedstawiają tę samą osobę.

        Args:
            img1_path: ścieżka do pierwszego zdjęcia (referencyjne)
            img2_path: ścieżka do drugiego zdjęcia (weryfikowane)

        Returns:
            VerificationResult z decyzją i similarity score
        """
        emb1, err1 = self._load_and_embed(img1_path)
        if err1:
            return VerificationResult(img1_path, img2_path, 0.0, False, self.threshold, err1)

        emb2, err2 = self._load_and_embed(img2_path)
        if err2:
            return VerificationResult(img1_path, img2_path, 0.0, False, self.threshold, err2)

        sim = cosine_similarity(emb1, emb2)
        decision = sim >= self.threshold

        return VerificationResult(
            img1_path=img1_path,
            img2_path=img2_path,
            similarity=sim,
            decision=decision,
            threshold=self.threshold,
        )

    def benchmark(self, pairs: list[tuple[str, str, bool]]) -> BiometricMetrics:
        """
        Ewaluuje system na zbiorze par zdjęć.

        Args:
            pairs: lista krotek (img1_path, img2_path, is_same_person)
                   is_same_person=True jeśli para przedstawia tę samą osobę

        Returns:
            BiometricMetrics z FAR, FRR, EER, AUC
        """
        genuine_scores = []
        impostor_scores = []
        errors = 0

        print(f"Benchmarking {len(pairs)} par...")
        for img1, img2, is_same in pairs:
            result = self.verify(img1, img2)

            if result.error:
                print(f"  POMINIĘTO: {result.error}")
                errors += 1
                continue

            if is_same:
                genuine_scores.append(result.similarity)
            else:
                impostor_scores.append(result.similarity)

        print(f"Przetworzono: {len(pairs)-errors}/{len(pairs)} par")
        print(f"  Genuine pairs: {len(genuine_scores)}")
        print(f"  Impostor pairs: {len(impostor_scores)}")

        if not genuine_scores or not impostor_scores:
            raise ValueError("Potrzebujesz par genuine i impostor do obliczenia metryk")

        metrics = compute_biometric_metrics(genuine_scores, impostor_scores)
        print_metrics_report(metrics)
        return metrics

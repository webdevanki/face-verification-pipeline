"""
Image preprocessing pipeline for biometric face verification.

Odpowiada za: wczytanie, walidację jakości i normalizację obrazów
przed przekazaniem do modelu detekcji twarzy.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageQualityReport:
    """Wynik kontroli jakości pojedynczego obrazu."""
    path: str
    is_valid: bool
    width: int
    height: int
    brightness: float       # średnia jasność 0-255
    blurriness: float       # wariancja laplasjanu - im wyższa tym ostrzejszy obraz
    reject_reason: Optional[str] = None


class ImagePreprocessor:
    """
    Pipeline wstępnego przetwarzania obrazów twarzy.

    W systemach biometrycznych jakość obrazu wejściowego
    bezpośrednio przekłada się na dokładność modelu.
    Śmieciowe dane wejściowe = złe embeddingi = wysokie FAR/FRR.
    """

    def __init__(
        self,
        min_size: int = 64,          # minimalny wymiar obrazu w pikselach
        max_size: int = 4096,        # maksymalny wymiar
        min_brightness: float = 30,  # zbyt ciemny obraz
        max_brightness: float = 220, # zbyt jasny / prześwietlony
        min_sharpness: float = 50,   # zbyt rozmyty
        target_size: tuple = (640, 640),  # rozmiar wyjściowy dla modelu
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.target_size = target_size

    def load(self, path: str | Path) -> Optional[np.ndarray]:
        """Wczytuje obraz. Zwraca None jeśli plik jest uszkodzony."""
        img = cv2.imread(str(path))
        if img is None:
            return None
        # OpenCV wczytuje w BGR, konwertujemy do RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def check_quality(self, img: np.ndarray, path: str = "") -> ImageQualityReport:
        """
        Sprawdza jakość obrazu i zwraca raport.

        Blurriness mierzymy wariancją laplasjanu:
        - Laplacjan wykrywa krawędzie
        - Ostry obraz = dużo wyraźnych krawędzi = wysoka wariancja
        - Rozmyty obraz = mało krawędzi = niska wariancja
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        brightness = float(np.mean(gray))
        blurriness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        reject_reason = None
        if w < self.min_size or h < self.min_size:
            reject_reason = f"za mały: {w}x{h}"
        elif w > self.max_size or h > self.max_size:
            reject_reason = f"za duży: {w}x{h}"
        elif brightness < self.min_brightness:
            reject_reason = f"za ciemny: brightness={brightness:.1f}"
        elif brightness > self.max_brightness:
            reject_reason = f"prześwietlony: brightness={brightness:.1f}"
        elif blurriness < self.min_sharpness:
            reject_reason = f"za rozmyty: sharpness={blurriness:.1f}"

        return ImageQualityReport(
            path=str(path),
            is_valid=reject_reason is None,
            width=w,
            height=h,
            brightness=brightness,
            blurriness=blurriness,
            reject_reason=reject_reason,
        )

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalizacja obrazu do formatu wejściowego modelu.

        Resize + normalizacja pikseli do zakresu [0, 1].
        Modele deep learning trenowane są na znormalizowanych danych.
        """
        resized = cv2.resize(img, self.target_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def process(self, path: str | Path) -> tuple[Optional[np.ndarray], ImageQualityReport]:
        """
        Pełny pipeline dla jednego obrazu.

        Returns:
            (przetworzone_zdjecie, raport_jakosci)
            Jeśli obraz jest odrzucony - zwraca (None, raport)
        """
        img = self.load(path)
        if img is None:
            return None, ImageQualityReport(
                path=str(path), is_valid=False,
                width=0, height=0, brightness=0, blurriness=0,
                reject_reason="nie można wczytać pliku"
            )

        report = self.check_quality(img, path)
        if not report.is_valid:
            return None, report

        processed = self.normalize(img)
        return processed, report

    def process_batch(self, paths: list[str | Path]) -> tuple[list, list[ImageQualityReport]]:
        """
        Przetwarza listę obrazów.

        Returns:
            (lista_obrazów, lista_raportów)
            Odrzucone obrazy są pomijane z listy obrazów, ale ich raporty są zachowane.
        """
        images = []
        reports = []

        for path in paths:
            img, report = self.process(path)
            reports.append(report)
            if img is not None:
                images.append(img)

        valid = sum(1 for r in reports if r.is_valid)
        print(f"Przetworzono: {valid}/{len(paths)} obrazów przeszło kontrolę jakości")

        return images, reports

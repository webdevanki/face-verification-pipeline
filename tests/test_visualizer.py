"""
Test wizualizacji - uruchom z głównego folderu:
    python tests/test_visualizer.py

Używa syntetycznych danych - nie potrzebujesz zdjęć.
Wygenerowane wykresy znajdziesz w experiments/
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.biometric import compute_biometric_metrics, print_metrics_report
from src.evaluation.visualizer import generate_full_report


def make_scores(n=300, seed=42):
    rng = np.random.default_rng(seed)
    genuine = rng.normal(loc=0.65, scale=0.12, size=n).clip(0, 1).tolist()
    impostor = rng.normal(loc=0.25, scale=0.10, size=n).clip(0, 1).tolist()
    return genuine, impostor


if __name__ == "__main__":
    print("=== Generowanie wizualizacji ===\n")

    genuine, impostor = make_scores(n=300)

    metrics = compute_biometric_metrics(genuine, impostor)
    print_metrics_report(metrics)

    generate_full_report(
        genuine_scores=genuine,
        impostor_scores=impostor,
        threshold=metrics.eer_threshold,
    )

    print("\nOtwórz folder experiments/ i sprawdź wygenerowane wykresy.")

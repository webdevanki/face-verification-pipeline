"""
Test metryk biometrycznych - uruchom z głównego folderu projektu:
    python tests/test_metrics.py

Ten test używa syntetycznych danych - nie potrzebujesz zdjęć.
Symulujemy wyniki systemu biometrycznego żeby sprawdzić logikę metryk.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.biometric import compute_biometric_metrics, print_metrics_report


def make_synthetic_scores(n=200, separation=0.3, seed=42):
    """
    Generuje syntetyczne scores symulujące działanie systemu biometrycznego.

    genuine scores: rozkład normalny wokół 0.7 (ta sama osoba - wysokie similarity)
    impostor scores: rozkład normalny wokół 0.2 (różne osoby - niskie similarity)

    Im większy separation, tym lepszy model (łatwiej odróżnić genuine od impostor).
    """
    rng = np.random.default_rng(seed)
    genuine = rng.normal(loc=0.7, scale=0.1, size=n).clip(0, 1)
    impostor = rng.normal(loc=0.2, scale=0.1, size=n).clip(0, 1)
    return genuine.tolist(), impostor.tolist()


def test_perfect_model():
    """Model idealny: genuine=1.0, impostor=0.0 → EER=0, AUC=1."""
    genuine = [1.0] * 100
    impostor = [0.0] * 100

    metrics = compute_biometric_metrics(genuine, impostor)

    assert metrics.auc_score == 1.0, f"AUC powinno być 1.0, got {metrics.auc_score}"
    assert metrics.eer < 0.01, f"EER powinno być ~0, got {metrics.eer}"
    print(f"OK  model idealny: AUC={metrics.auc_score:.4f}, EER={metrics.eer*100:.2f}%")


def test_random_model():
    """Model losowy: genuine i impostor z tego samego rozkładu → EER≈50%, AUC≈0.5."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 200).tolist()
    genuine = scores[:100]
    impostor = scores[100:]

    metrics = compute_biometric_metrics(genuine, impostor)

    assert 0.3 < metrics.auc_score < 0.7, f"AUC losowego modelu ~0.5, got {metrics.auc_score}"
    print(f"OK  model losowy:  AUC={metrics.auc_score:.4f}, EER={metrics.eer*100:.2f}%")


def test_good_model():
    """Dobry model: wyraźna separacja genuine/impostor → niski EER."""
    genuine, impostor = make_synthetic_scores(n=200, separation=0.3)

    metrics = compute_biometric_metrics(genuine, impostor)
    print_metrics_report(metrics)

    assert metrics.auc_score > 0.9, f"AUC powinno być >0.9, got {metrics.auc_score}"
    assert metrics.eer < 0.15, f"EER powinno być <15%, got {metrics.eer*100:.2f}%"
    print(f"OK  dobry model:   AUC={metrics.auc_score:.4f}, EER={metrics.eer*100:.2f}%")


def test_custom_threshold():
    """Sprawdź jak zmiana progu wpływa na FAR i FRR - trade-off."""
    genuine, impostor = make_synthetic_scores(n=500)

    print("\nTrade-off FAR vs FRR przy różnych progach:")
    print(f"{'Próg':>8} {'FAR':>8} {'FRR':>8}")
    print("-" * 28)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = compute_biometric_metrics(genuine, impostor, threshold=threshold)
        marker = " ← EER" if abs(metrics.far - metrics.frr) < 0.02 else ""
        print(f"{threshold:>8.1f} {metrics.far*100:>7.1f}% {metrics.frr*100:>7.1f}%{marker}")


if __name__ == "__main__":
    print("=== Test metryk biometrycznych ===\n")
    test_perfect_model()
    test_random_model()
    test_good_model()
    test_custom_threshold()
    print("\nWszystkie testy przeszły!")

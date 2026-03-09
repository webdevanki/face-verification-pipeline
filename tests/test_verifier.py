"""
Test end-to-end weryfikacji twarzy - uruchom z głównego folderu:
    python tests/test_verifier.py

Potrzebujesz:
    data/raw/test.jpg   - zdjęcie osoby A
    data/raw/test2.jpg  - drugie zdjęcie osoby A (genuine pair)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.verifier import FaceVerifier


def test_single_verification():
    verifier = FaceVerifier(threshold=0.45)

    print("--- Weryfikacja: to samo zdjęcie (powinno być MATCH) ---")
    result = verifier.verify("data/raw/test.jpg", "data/raw/test.jpg")
    if result.error:
        print(f"BŁĄD: {result.error}")
        return

    print(f"Similarity:  {result.similarity:.4f}")
    print(f"Threshold:   {result.threshold}")
    print(f"Decyzja:     {'MATCH ✓' if result.decision else 'NO MATCH ✗'}")
    assert result.decision, "To samo zdjęcie powinno dać MATCH"
    print("OK\n")

    if Path("data/raw/test2.jpg").exists():
        print("--- Weryfikacja: dwa różne zdjęcia ---")
        result2 = verifier.verify("data/raw/test.jpg", "data/raw/test2.jpg")
        print(f"Similarity:  {result2.similarity:.4f}")
        print(f"Threshold:   {result2.threshold}")
        print(f"Decyzja:     {'MATCH ✓' if result2.decision else 'NO MATCH ✗'}")
        print(f"\nJeśli to ta sama osoba i decyzja to NO MATCH,")
        print(f"spróbuj obniżyć próg: FaceVerifier(threshold={result2.similarity - 0.05:.2f})")


if __name__ == "__main__":
    print("=== Test weryfikacji end-to-end ===\n")
    test_single_verification()

"""
Metryki biometryczne: FAR, FRR, EER, ROC.

To jest najważniejszy moduł w systemach biometrycznych.
Odpowiada na pytanie: "jak dobry jest nasz model i jaki próg ustawić?"
"""

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_curve, auc


@dataclass
class BiometricMetrics:
    """Wynik ewaluacji systemu biometrycznego."""
    threshold: float        # próg decyzyjny
    far: float              # False Acceptance Rate
    frr: float              # False Rejection Rate
    eer: float              # Equal Error Rate
    eer_threshold: float    # próg przy którym FAR == FRR
    auc_score: float        # Area Under ROC Curve (1.0 = idealny model)


def compute_biometric_metrics(
    genuine_scores: list[float],    # similarity scores: ta sama osoba
    impostor_scores: list[float],   # similarity scores: różne osoby
    threshold: float = None,
) -> BiometricMetrics:
    """
    Oblicza metryki biometryczne na podstawie scores.

    Jak to działa:
    - genuine_scores: wyniki porównania par "ta sama osoba" (powinny być wysokie)
    - impostor_scores: wyniki porównania par "różne osoby" (powinny być niskie)
    - Na podstawie rozkładu tych scores wybieramy optymalny próg

    Args:
        genuine_scores: lista similarity scores dla par tej samej osoby
        impostor_scores: lista similarity scores dla par różnych osób
        threshold: próg decyzyjny. Jeśli None - automatycznie wyznaczamy EER threshold

    Returns:
        BiometricMetrics z FAR, FRR, EER i AUC
    """
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    # budujemy etykiety: 1 = ta sama osoba, 0 = różne osoby
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])

    # ROC curve z sklearn
    # fpr = False Positive Rate = FAR w biometrii
    # tpr = True Positive Rate = 1 - FRR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    # EER: punkt gdzie FAR == FRR, czyli fpr == 1 - tpr
    frr_curve = 1 - tpr
    diff = np.abs(fpr - frr_curve)
    eer_idx = np.argmin(diff)
    eer = float((fpr[eer_idx] + frr_curve[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    # jeśli nie podano progu, użyj EER threshold
    if threshold is None:
        threshold = eer_threshold

    # oblicz FAR i FRR dla wybranego progu
    far = float(np.mean(impostor >= threshold))   # impostors zaakceptowani
    frr = float(np.mean(genuine < threshold))      # genuine odrzuceni

    return BiometricMetrics(
        threshold=threshold,
        far=far,
        frr=frr,
        eer=eer,
        eer_threshold=eer_threshold,
        auc_score=auc_score,
    )


def print_metrics_report(metrics: BiometricMetrics):
    """Wypisuje czytelny raport metryk."""
    print("\n" + "="*50)
    print("RAPORT METRYK BIOMETRYCZNYCH")
    print("="*50)
    print(f"AUC Score:       {metrics.auc_score:.4f}  (1.0 = idealny)")
    print(f"EER:             {metrics.eer*100:.2f}%    (im niższy tym lepszy)")
    print(f"EER Threshold:   {metrics.eer_threshold:.4f}")
    print("-"*50)
    print(f"Próg decyzyjny:  {metrics.threshold:.4f}")
    print(f"FAR:             {metrics.far*100:.2f}%    (fałszywe akceptacje)")
    print(f"FRR:             {metrics.frr*100:.2f}%    (fałszywe odrzucenia)")
    print("="*50)
    print()
    if metrics.eer < 0.05:
        print("Model: DOSKONAŁY (EER < 5%)")
    elif metrics.eer < 0.10:
        print("Model: DOBRY (EER < 10%)")
    elif metrics.eer < 0.20:
        print("Model: PRZECIĘTNY (EER < 20%)")
    else:
        print("Model: SŁABY (EER > 20%) - sprawdź jakość danych")

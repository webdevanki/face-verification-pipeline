"""
Wizualizacje metryk biometrycznych.

Generuje wykresy ROC curve, FAR/FRR trade-off i score distributions
- standardowe narzędzia raportowania w projektach R&D.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(
    genuine_scores: list[float],
    impostor_scores: list[float],
    output_path: str = "experiments/roc_curve.png",
    title: str = "ROC Curve - Face Verification System",
):
    """
    Rysuje krzywą ROC z zaznaczonym punktem EER.

    ROC (Receiver Operating Characteristic) pokazuje trade-off między
    TPR (czułość) a FPR (1 - specyficzność) dla wszystkich możliwych progów.
    Im bardziej wykres "wybrzusza się" w lewy górny róg, tym lepszy model.
    """
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    # znajdź punkt EER
    frr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - frr))
    eer = (fpr[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    # krzywa ROC
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")

    # linia losowego modelu
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Model losowy (AUC = 0.5)")

    # punkt EER
    ax.scatter(
        fpr[eer_idx], tpr[eer_idx],
        color="red", s=100, zorder=5,
        label=f"EER = {eer*100:.2f}% (threshold = {eer_threshold:.3f})"
    )

    ax.set_xlabel("FAR - False Acceptance Rate (FPR)", fontsize=12)
    ax.set_ylabel("1 - FRR - True Acceptance Rate (TPR)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve zapisana: {output_path}")


def plot_score_distribution(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float,
    output_path: str = "experiments/score_distribution.png",
    title: str = "Score Distribution - Genuine vs Impostor",
):
    """
    Histogram rozkładu scores genuine i impostor.

    Idealny system: dwa rozłączne klastry bez nakładania.
    Nakładanie się rozkładów = strefa błędów = EER.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 1, 50)

    ax.hist(genuine_scores, bins=bins, alpha=0.6, color="green",
            label=f"Genuine (ta sama osoba) n={len(genuine_scores)}")
    ax.hist(impostor_scores, bins=bins, alpha=0.6, color="red",
            label=f"Impostor (różne osoby) n={len(impostor_scores)}")

    ax.axvline(x=threshold, color="black", linestyle="--", lw=2,
               label=f"Próg decyzyjny = {threshold:.3f}")

    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Liczba par", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Score distribution zapisana: {output_path}")


def plot_far_frr_tradeoff(
    genuine_scores: list[float],
    impostor_scores: list[float],
    output_path: str = "experiments/far_frr_tradeoff.png",
):
    """
    Wykres FAR i FRR w funkcji progu decyzyjnego.

    Przecięcie krzywych FAR i FRR = EER = optymalny próg.
    """
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    thresholds = np.linspace(0, 1, 200)
    far_values = [np.mean(impostor >= t) for t in thresholds]
    frr_values = [np.mean(genuine < t) for t in thresholds]

    # EER: próg gdzie FAR ≈ FRR
    diff = np.abs(np.array(far_values) - np.array(frr_values))
    eer_idx = np.argmin(diff)
    eer_threshold = thresholds[eer_idx]
    eer = (far_values[eer_idx] + frr_values[eer_idx]) / 2

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, far_values, color="red", lw=2, label="FAR (False Acceptance Rate)")
    ax.plot(thresholds, frr_values, color="blue", lw=2, label="FRR (False Rejection Rate)")

    ax.axvline(x=eer_threshold, color="black", linestyle="--", lw=1.5,
               label=f"EER = {eer*100:.2f}% @ threshold={eer_threshold:.3f}")
    ax.scatter([eer_threshold], [eer], color="black", s=80, zorder=5)

    ax.set_xlabel("Próg decyzyjny (threshold)", fontsize=12)
    ax.set_ylabel("Współczynnik błędu", fontsize=12)
    ax.set_title("FAR / FRR Trade-off", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"FAR/FRR trade-off zapisany: {output_path}")


def generate_full_report(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float,
    output_dir: str = "experiments",
):
    """Generuje wszystkie wykresy naraz."""
    print("\nGenerowanie raportów wizualnych...")
    plot_roc_curve(genuine_scores, impostor_scores,
                   output_path=f"{output_dir}/roc_curve.png")
    plot_score_distribution(genuine_scores, impostor_scores, threshold,
                            output_path=f"{output_dir}/score_distribution.png")
    plot_far_frr_tradeoff(genuine_scores, impostor_scores,
                          output_path=f"{output_dir}/far_frr_tradeoff.png")
    print(f"\nWszystkie wykresy zapisane w: {output_dir}/")

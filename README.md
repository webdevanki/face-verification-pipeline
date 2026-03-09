# Face Verification Pipeline

End-to-end system weryfikacji tożsamości oparty na biometrii twarzy.
Projekt demonstruje pełny pipeline R&D: od surowych obrazów do ewaluacji modelu z metrykami biometrycznymi.

## Architektura systemu

```
Obraz wejściowy
      │
      ▼
┌─────────────────┐
│  Preprocessor   │  kontrola jakości (jasność, ostrość, rozmiar)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Face Detector  │  RetinaFace – detekcja i wycinanie twarzy (112x112)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Face Embedder  │  ArcFace – ekstrakcja wektora cech (512-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Verifier       │  cosine similarity + próg decyzyjny
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Metrics        │  FAR / FRR / EER / ROC / AUC
└─────────────────┘
```

## Wyniki

| Metryka | Wartość |
|---------|---------|
| AUC | 0.9976 |
| EER | 2.83% |
| Optymalny próg | 0.444 |
| FAR @ EER | 2.83% |
| FRR @ EER | 2.83% |

### ROC Curve
![ROC Curve](experiments/roc_curve.png)

### Score Distribution
![Score Distribution](experiments/score_distribution.png)

### FAR/FRR Trade-off
![FAR/FRR](experiments/far_frr_tradeoff.png)

## Stack technologiczny

| Komponent | Technologia |
|-----------|-------------|
| Detekcja twarzy | RetinaFace (InsightFace) |
| Embeddingi | ArcFace w600k_mbf (512-dim) |
| Inference | ONNX Runtime (CPU) |
| Metryki | scikit-learn (ROC, AUC) |
| Wizualizacje | matplotlib |

## Struktura projektu

```
├── src/
│   ├── pipeline/
│   │   ├── preprocessor.py   # kontrola jakości obrazów
│   │   ├── detector.py       # detekcja i wycinanie twarzy
│   │   └── embedder.py       # ekstrakcja embeddingów ArcFace
│   ├── metrics/
│   │   └── biometric.py      # FAR, FRR, EER, ROC, AUC
│   └── evaluation/
│       ├── verifier.py       # end-to-end weryfikacja 1:1
│       └── visualizer.py     # wykresy i raporty
└── tests/
    ├── test_preprocessor.py
    ├── test_detector.py
    ├── test_embedder.py
    ├── test_metrics.py
    └── test_verifier.py
```

## Instalacja

```bash
conda create -n face-verification python=3.10 -y
conda activate face-verification
pip install opencv-python numpy pillow onnxruntime insightface scikit-learn matplotlib pandas tqdm
```

## Uruchomienie

```bash
# weryfikacja pary zdjęć
python tests/test_verifier.py

# metryki biometryczne
python tests/test_metrics.py

# generowanie wykresów
python tests/test_visualizer.py
```

## Kluczowe koncepty

**Weryfikacja vs identyfikacja**
System implementuje weryfikację 1:1 - odpowiada na pytanie "czy to ta sama osoba co na referencyjnym zdjęciu?", nie "kto to jest?".

**Embeddingi**
ArcFace zamienia twarz w wektor 512 liczb. Dwie twarze tej samej osoby dają podobne wektory (cosine similarity > 0.44), twarze różnych osób - odległe.

**FAR / FRR trade-off**
Obniżenie progu decyzyjnego zmniejsza FRR (mniej odrzuconych prawdziwych użytkowników) ale zwiększa FAR (więcej fałszywych akceptacji). EER to punkt optymalnego kompromisu.

**Jakość danych**
Kontrola jakości wejściowej (ostrość laplasjanu, jasność) bezpośrednio przekłada się na jakość embeddingów. Złej jakości dane wejściowe = wyższy EER.

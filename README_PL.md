# FingerSight

Aplikacja webowa do rozpoznawania **Polskiego Alfabetu Palcowego (PAP)** w czasie rzeczywistym z kamery. Dłoń wykrywa MediaPipe w przeglądarce, landmarki trafiają do backendu FastAPI z modelem RandomForest.

---

## Stos technologiczny

| Warstwa | Technologia |
|---|---|
| Frontend | React 19 + Vite |
| Detekcja dłoni | MediaPipe Tasks-Vision (HandLandmarker) |
| Backend | FastAPI + Uvicorn (Python 3.11+) |
| Model ML | scikit-learn RandomForest |
| Komunikacja | REST API (`POST /api/predict`) |

---

## Uruchomienie lokalne

### 1. Backend

```bash
cd backend

# Utwórz środowisko wirtualne i zainstaluj zależności
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Uruchom serwer
uvicorn app.main:app --reload
```

API dostępne na `http://localhost:8000`  
Dokumentacja (Swagger): `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Aplikacja dostępna na `http://localhost:5173`

> **Ważne:** Backend musi działać przed uruchomieniem frontendu. Frontend ma fallback klasyfikator geometryczny (5 liter), ale pełne rozpoznawanie wymaga API.

---

## API

```
GET  /api/ping     → {"status": "ok"}
POST /api/predict  → {"letter": "A", "confidence": 0.97}
GET  /api/history  → {"history": ["A", "B", "M"]}
```

**Request `/api/predict`:**
```json
{
  "landmarks": [[x, y, z], ...]
}
```
21 punktów dłoni (x, y, z) z MediaPipe. Zwraca `{"letter": null, "confidence": 0.0}` gdy pewność < 0.6.

---

## Aktualny stan modelu

Model wytrenowany jest na **16 literach**: `A B C E I L M N O P R S T U W Y`

Dane zebrane jedną ręką, przy jednym oświetleniu — model działa, ale ma ograniczoną dokładność w innych warunkach. Szczegóły treningu poniżej.

---

## Jak poprawić model

### Problem: skąd słaba accuracy?

Model osiągnął 100% na danych testowych, ale to dane z tej samej sesji — ta sama ręka, to samo oświetlenie, ten sam kąt. W praktyce accuracy spada bo:

- Inne oświetlenie → inne wartości landmarków
- Inny kąt dłoni → inne proporcje
- Inna ręka (inny użytkownik) → inne rozmiary

### Co zrobić

**1. Zbierz więcej próbek — z różnorodnych pozycji**

```bash
cd backend
venv\Scripts\activate

# Zbierz 200 próbek dla litery A z nowego kąta / oświetlenia
python train/collect.py --letter A --samples 200 --camera 0

# Collect.py przy pierwszym uruchomieniu pobierze model MediaPipe (~25 MB)
# SPACJA = start/stop nagrywania, Q = zakończ
```

Rób każdą literę **minimum 3 razy**: raz normalnie, raz z dłonią wyżej, raz z boku. Różnorodność > ilość.

**2. Retrenuj model**

```bash
python train/train.py
```

**3. Sprawdź accuracy**

```bash
python train/evaluate.py
# Generuje classification_report + confusion_matrix.png
```

Celuj w **>90% accuracy**. Jeśli któraś litera ma recall < 0.8 — zbierz dla niej więcej próbek.

**4. Dodaj brakujące litery statyczne**

Aplikacja aktualnie obsługuje tylko **litery statyczne PAP** — takie, które mają stały kształt dłoni bez ruchu. Litery z ogonkami (Ą, Ę, Ó, Ń, Ć, Ś, Ź, Ż, Ł) wymagają ruchu dłoni — rozpoznawanie sekwencji klatek nie jest jeszcze zaimplementowane.

```bash
python train/collect.py --letter D --samples 300 --camera 0
python train/collect.py --letter F --samples 300 --camera 0
# ... itd.
python train/train.py
```

**5. Praktyczne wskazówki podczas zbierania danych**

- Jednolite tło (biała ściana) — MediaPipe działa lepiej
- Dobre, równomierne oświetlenie — bez silnego podświetlenia z tyłu
- Dłoń w odległości 30–60 cm od kamery
- Zbieraj próbki od różnych osób jeśli możliwe

---

## Zmienne środowiskowe

Stwórz plik `backend/.env` (szablon w `backend/.env.example`):

```env
MODEL_PATH=app/models/model.pkl
HISTORY_MAX_LENGTH=50
CONFIDENCE_THRESHOLD=0.6
```

`CONFIDENCE_THRESHOLD` — jeśli model zwraca za dużo `null`, obniż do `0.5`. Jeśli za dużo błędów — podnieś do `0.7`.

---

## Struktura projektu

```
finger-sight/
├── frontend/
│   └── src/
│       ├── App.jsx              # główny komponent UI
│       ├── App.css              # style
│       ├── classifier.js        # fallback klasyfikator geometryczny
│       ├── hooks/
│       │   └── useHandDetector.js  # MediaPipe + komunikacja z API
│       └── api/
│           └── predict.js       # fetch wrapper
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── classifier.py        # HandClassifier (RandomForest)
│   │   ├── normalizer.py        # normalizacja landmarków
│   │   ├── config.py            # zmienne środowiskowe
│   │   └── models/
│   │       └── model.pkl        # wytrenowany model (w repo)
│   ├── data/                    # landmarks.csv (w .gitignore)
│   ├── train/
│   │   ├── collect.py           # zbieranie danych z kamery
│   │   ├── train.py             # trening modelu
│   │   └── evaluate.py          # raport accuracy + confusion matrix
│   └── requirements.txt
└── README.md
```

---

## Zespół

- **Frontend** — React, MediaPipe JS, UI/UX
- **Backend/ML** — FastAPI, model RandomForest, zbieranie danych

# FingerSight 🖐️

Aplikacja webowa do rozpoznawania **polskiego alfabetu palcowego (PAP)** w czasie rzeczywistym z kamery.

## Co robi

Użytkownik pokazuje dłoń do kamery, aplikacja wykrywa gest i wyświetla odpowiadającą mu literę polskiego alfabetu (A–Ż). Działa w przeglądarce bez instalacji.

## Stos technologiczny

| Warstwa | Technologia |
|---|---|
| Frontend | React 18 + Vite |
| Detekcja dłoni | MediaPipe Tasks-Vision (HandLandmarker) |
| Backend | FastAPI + Uvicorn (Python 3.11) |
| Model ML | scikit-learn (RandomForest) / Keras (MLP) |
| Komunikacja | REST API (POST /api/predict) |

## Uruchomienie lokalne

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Aplikacja dostępna na `http://localhost:5173`

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API dostępne na `http://localhost:8000`  
Dokumentacja: `http://localhost:8000/docs`

## Struktura projektu

```
fingersight/
├── frontend/               # React app
│   └── src/
│       ├── components/     # UI: CameraView, LetterDisplay, History
│       ├── hooks/          # useHandDetector, usePrediction
│       └── api/            # fetch wrappers do backendu
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI endpoints
│   │   ├── classifier.py   # HandClassifier (RF / Keras)
│   │   ├── normalizer.py   # normalizacja landmarków
│   │   └── models/         # zapisane modele (.pkl, .keras)
│   ├── data/               # CSV z landmarkami (zbierane lokalnie)
│   └── train/              # skrypty treningowe
├── .github/
│   ├── copilot-instructions.md
│   └── prompts/
│       └── collect-data.prompt.md
├── AGENTS.md
├── CLAUDE.md
└── README.md
```

## API

```
GET  /api/ping       → {"status": "ok"}
POST /api/predict    → {"letter": "A", "confidence": 0.97}
GET  /api/history    → {"history": ["A", "B", "M"]}
```

**Format wejścia `/api/predict`:**
```json
{
  "landmarks": [
    [0.51, 0.73, 0.002],
    ...
  ]
}
```
21 punktów dłoni (x, y, z) wyznaczonych przez MediaPipe.

## Litery (32 klasy)

`A Ą B C Ć D E Ę F G H I J K L Ł M N Ń O Ó P R S Ś T U W Y Z Ź Ż`

## Zespół

- **Frontend** — React, MediaPipe JS, UI
- **Backend/ML** — FastAPI, model, Docker
- **Integracja** — połączenie front/back, testy, CI

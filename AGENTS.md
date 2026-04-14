# AGENTS.md — FingerSight

Ten plik jest przeznaczony dla agentów AI (GitHub Copilot, Claude Code, itp.).
Zawiera pełny kontekst projektu, konwencje i zasady których należy przestrzegać.

---

## Cel projektu

Aplikacja webowa rozpoznająca **polski alfabet palcowy (PAP)** z kamery w czasie rzeczywistym.
Pipeline: kamera → MediaPipe (detekcja dłoni, 21 punktów) → FastAPI (klasyfikacja ML) → wyświetlenie litery.

## Architektura

```
Przeglądarka
  └── React App
        ├── react-webcam         (strumień wideo)
        ├── MediaPipe HandLandmarker (21 landmarks × [x,y,z] = 63 cechy)
        └── fetch POST /api/predict → FastAPI Backend
                                          └── HandClassifier.predict()
                                                └── RandomForest / Keras MLP
                                                      └── {"letter": "A", "confidence": 0.97}
```

## Struktura plików

```
frontend/src/
  components/
    CameraView.jsx        # strumień wideo + canvas z landmarkami
    LetterDisplay.jsx     # duża litera + pasek pewności
    HistoryPanel.jsx      # lista ostatnich rozpoznanych liter
  hooks/
    useHandDetector.js    # inicjalizacja MediaPipe, zwraca landmarks co klatkę
    usePrediction.js      # wysyła landmarks do API, zwraca {letter, confidence}
  api/
    predict.js            # fetch wrapper dla POST /api/predict

backend/app/
  main.py                 # FastAPI app, definicja endpointów
  classifier.py           # klasa HandClassifier
  normalizer.py           # funkcje normalizacji landmarków
  models/                 # zapisane modele: model.pkl lub model.keras
  config.py               # ścieżki, zmienne środowiskowe

backend/data/
  landmarks.csv           # zebrane dane treningowe (label + 63 cechy)

backend/train/
  collect.py              # skrypt do zbierania danych z kamery
  train.py                # trenowanie i zapis modelu
  evaluate.py             # raport: accuracy, confusion matrix
```

## Konwencje kodu

### Python (backend)
- Python 3.11, type hints wszędzie, docstringi na każdej funkcji/klasie
- PEP8, formatowanie przez `black`
- Pydantic do walidacji requestów i responsów
- Zmienne środowiskowe przez `python-dotenv` (plik `.env`, nigdy hardcode)
- Błędy: zawsze `HTTPException(status_code=..., detail="czytelny opis")`

### React (frontend)
- Tylko komponenty funkcyjne + hooks
- Nazwy komponentów: PascalCase (`CameraView`, nie `camera-view`)
- Nazwy hooków: camelCase z prefiksem `use` (`useHandDetector`)
- Brak Redux — lokalny stan przez `useState`/`useReducer`
- Komunikacja z API wyłącznie przez wrappery z `src/api/`

### Git
- Commity po angielsku, format: `feat:`, `fix:`, `chore:`, `docs:`
- Branching: `main` (stabilny), `dev` (bieżący), feature branche od `dev`

## Kluczowe klasy i interfejsy

### HandClassifier (backend/app/classifier.py)
```python
class HandClassifier:
    def train(self, X: list[list[float]], y: list[str]) -> None: ...
    def predict(self, landmarks: list[list[float]]) -> tuple[str, float]:
        """Zwraca (litera, pewność) np. ('A', 0.97)"""
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### LandmarkNormalizer (backend/app/normalizer.py)
```python
def normalize(landmarks: list[list[float]]) -> list[float]:
    """
    Przesuwa środek do nadgarstka (punkt 0),
    skaluje przez odległość nadgarstek-środkowy palec,
    zwraca spłaszczoną listę 63 cech.
    """
```

## API — pełna specyfikacja

### GET /api/ping
```json
Response: {"status": "ok"}
```

### POST /api/predict
```json
Request:
{
  "landmarks": [[x, y, z], [x, y, z], ...]   // dokładnie 21 punktów
}

Response (sukces):
{"letter": "A", "confidence": 0.97}

Response (brak pewności):
{"letter": null, "confidence": 0.0}

Błędy:
400 — nieprawidłowa liczba punktów (≠21)
422 — błędny format JSON (Pydantic)
500 — błąd wewnętrzny serwera
```

### GET /api/history
```json
Response: {"history": ["A", "B", "M", "Ż"]}
```

## Format danych treningowych (landmarks.csv)

```
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
A,0.51,0.73,0.002,...
```
- 1 kolumna label + 63 kolumny cech = 64 kolumny łącznie
- Cechy **już znormalizowane** względem nadgarstka

## Litery (32 klasy)

```
A Ą B C Ć D E Ę F G H I J K L Ł M N Ń O Ó P R S Ś T U W Y Z Ź Ż
```

## Rzeczy których NIE robić

- Nie używaj `create-react-app` — projekt używa Vite
- Nie commituj pliku `.env`, `models/`, `data/landmarks.csv` — są w `.gitignore`
- Nie wysyłaj klatek wideo do backendu — tylko landmarki (63 liczby)
- Nie używaj `any` w type hints Pythona
- Nie modyfikuj `landmarks.csv` ręcznie — tylko przez `collect.py`
- Nie uruchamiaj MediaPipe po stronie backendu w MVP — tylko w przeglądarce (JS)

## Zmienne środowiskowe

```
# backend/.env
MODEL_PATH=app/models/model.pkl
HISTORY_MAX_LENGTH=50

# frontend/.env
VITE_API_URL=http://localhost:8000
```

## Zależności

### backend/requirements.txt
```
fastapi==0.111.0
uvicorn==0.30.0
pydantic==2.7.0
scikit-learn==1.5.0
tensorflow==2.16.0
mediapipe==0.10.14
opencv-python==4.9.0.80
numpy==1.26.4
python-dotenv==1.0.1
```

### frontend (package.json devDependencies)
```
vite, @vitejs/plugin-react
@mediapipe/tasks-vision
react-webcam
@mui/material @emotion/react @emotion/styled
```

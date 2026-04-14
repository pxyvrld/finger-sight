# CLAUDE.md — FingerSight

Plik dla Claude Code. Zawiera instrukcje specyficzne dla tego projektu.

## Czym jest ten projekt

FingerSight to webowa aplikacja do rozpoznawania polskiego alfabetu palcowego (PAP).
Szczegółowa architektura i konwencje są w `AGENTS.md` — przeczytaj go jako pierwsze.

## Jak uruchomić projekt lokalnie

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend (osobny terminal)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Najważniejsze pliki do których będziesz sięgać

| Plik | Co robi |
|---|---|
| `backend/app/main.py` | Endpointy FastAPI |
| `backend/app/classifier.py` | Logika modelu ML |
| `backend/app/normalizer.py` | Normalizacja landmarków |
| `frontend/src/hooks/useHandDetector.js` | MediaPipe w React |
| `frontend/src/hooks/usePrediction.js` | Komunikacja z API |
| `backend/data/landmarks.csv` | Dane treningowe |
| `backend/train/train.py` | Skrypt treningowy |

## Typowe zadania i jak je wykonać

### Dodanie nowego endpointu
1. Dodaj model Pydantic w `backend/app/main.py`
2. Dodaj endpoint z dekoratorem `@app.post(...)` lub `@app.get(...)`
3. Dodaj fetch wrapper w `frontend/src/api/`

### Trening modelu
```bash
cd backend
python train/collect.py --letter A    # zbierz dane dla litery A
python train/train.py                 # trenuj model na landmarks.csv
python train/evaluate.py              # sprawdź accuracy i confusion matrix
```

### Dodanie nowego komponentu React
Utwórz plik w `frontend/src/components/NazwaKomponentu.jsx`.
Używaj tylko hooków, bez klas. Eksportuj jako default export.

## Zasady których przestrzegamy

- Landmarki są **zawsze normalizowane** przed klasyfikacją (patrz `normalizer.py`)
- Model jest ładowany **raz przy starcie serwera**, nie przy każdym requeście
- Frontend wysyła landmarki **co 200ms** (nie każdą klatkę) — debouncing w `usePrediction.js`
- Confidence poniżej `0.6` → zwracamy `{"letter": null}`, frontend pokazuje `?`

## Czego nie zmieniać bez konsultacji z zespołem

- Format CSV w `data/landmarks.csv` (zmiana złamie istniejące dane)
- Kontrakt API `/api/predict` (request/response schema)
- Kolejność punktów landmarków (MediaPipe definiuje indeksy 0–20)

## Przydatne komendy

```bash
# Sprawdź API ręcznie
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[0.5,0.5,0],[0.5,0.6,0],[0.5,0.7,0],[0.5,0.8,0],[0.5,0.9,0],[0.6,0.5,0],[0.6,0.6,0],[0.6,0.7,0],[0.6,0.8,0],[0.7,0.5,0],[0.7,0.6,0],[0.7,0.7,0],[0.7,0.8,0],[0.8,0.5,0],[0.8,0.6,0],[0.8,0.7,0],[0.8,0.8,0],[0.9,0.5,0],[0.9,0.6,0],[0.9,0.7,0],[0.9,0.8,0]]}'

# Formatowanie kodu Python
cd backend && black app/ train/

# Uruchom z logami
uvicorn app.main:app --reload --log-level debug
```

## Status MVP

- [ ] Frontend: kamera + MediaPipe (landmarki widoczne na canvas)
- [ ] Backend: dummy endpoint /api/predict (zwraca zawsze "A")
- [ ] Połączenie front-back (fetch działa end-to-end)
- [ ] Zbieranie danych (collect.py)
- [ ] Trening modelu (RandomForest, >90% accuracy)
- [ ] Integracja modelu z API
- [ ] UI: historia liter, wskaźnik pewności
- [ ] Testy z różnymi osobami

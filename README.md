# FingerSight

A web application for real-time recognition of the **Polish Finger Alphabet (PAP)** using a camera. Hand detection runs in the browser via MediaPipe; landmarks are sent to a FastAPI backend with a RandomForest classifier.

> Polski opis: [README_PL.md](README_PL.md)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + Vite |
| Hand detection | MediaPipe Tasks-Vision (HandLandmarker) |
| Backend | FastAPI + Uvicorn (Python 3.11+) |
| ML model | scikit-learn RandomForest |
| Communication | REST API (`POST /api/predict`) |

---

## Running locally

### 1. Backend

```bash
cd backend

# Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:5173`

> **Important:** The backend must be running before starting the frontend. The frontend has a built-in geometric fallback classifier (5 letters), but full recognition requires the API.

---

## API

```
GET  /api/ping     → {"status": "ok"}
POST /api/predict  → {"letter": "A", "confidence": 0.97}
GET  /api/history  → {"history": ["A", "B", "M"]}
```

**Request body for `/api/predict`:**
```json
{
  "landmarks": [[x, y, z], ...]
}
```
21 hand points (x, y, z) from MediaPipe. Returns `{"letter": null, "confidence": 0.0}` when confidence is below 0.6.

---

## Current model state

The model is trained on **16 letters**: `A B C E I L M N O P R S T U W Y`

Data was collected with one hand under one lighting condition — the model works, but accuracy drops under different conditions. See below for how to improve it.

---

## How to improve the model

### Why is accuracy limited?

The model reached 100% on test data, but those samples came from the same session — same hand, same lighting, same angle. In practice accuracy drops because:

- Different lighting → different landmark values
- Different hand angle → different proportions
- Different user's hand → different sizes

### What to do

**1. Collect more diverse samples**

```bash
cd backend
venv\Scripts\activate

# Collect 200 samples for letter A from a new angle / lighting
python train/collect.py --letter A --samples 200 --camera 0

# First run downloads the MediaPipe model (~25 MB)
# SPACE = start/stop recording, Q = quit
```

Collect each letter **at least 3 times**: once normally, once with hand higher, once from the side. Diversity beats quantity.

**2. Retrain the model**

```bash
python train/train.py
```

**3. Check accuracy**

```bash
python train/evaluate.py
# Generates a classification_report + confusion_matrix.png
```

Aim for **>90% accuracy**. If any letter has recall < 0.8 — collect more samples for it.

**4. Add missing static letters**

The app currently supports only **static PAP letters** — those with a fixed hand shape and no movement. Letters with diacritics (Ą, Ę, Ó, Ń, Ć, Ś, Ź, Ż, Ł) require hand motion — sequence-based recognition is not yet implemented.

You can add missing static letters (e.g. `D F G H J K Z`):

```bash
python train/collect.py --letter D --samples 300 --camera 0
python train/collect.py --letter F --samples 300 --camera 0
# ... etc.
python train/train.py
```

**5. Tips for collecting good data**

- Plain background (white wall) — MediaPipe works better
- Good, even lighting — avoid strong backlighting
- Hand 30–60 cm from the camera
- Collect samples from multiple people if possible

---

## Environment variables

Create `backend/.env` (template in `backend/.env.example`):

```env
MODEL_PATH=app/models/model.pkl
HISTORY_MAX_LENGTH=50
CONFIDENCE_THRESHOLD=0.6
```

`CONFIDENCE_THRESHOLD` — if the model returns too many `null` results, lower it to `0.5`. If too many wrong predictions — raise it to `0.7`.

---

## Project structure

```
finger-sight/
├── frontend/
│   └── src/
│       ├── App.jsx              # main UI component
│       ├── App.css              # styles
│       ├── classifier.js        # geometric fallback classifier
│       ├── hooks/
│       │   └── useHandDetector.js  # MediaPipe + API communication
│       └── api/
│           └── predict.js       # fetch wrapper
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── classifier.py        # HandClassifier (RandomForest)
│   │   ├── normalizer.py        # landmark normalization
│   │   ├── config.py            # environment variables
│   │   └── models/
│   │       └── model.pkl        # trained model (included in repo)
│   ├── data/                    # landmarks.csv (in .gitignore)
│   ├── train/
│   │   ├── collect.py           # data collection from webcam
│   │   ├── train.py             # model training
│   │   └── evaluate.py          # accuracy report + confusion matrix
│   └── requirements.txt
└── README.md
```

---

## Team

- **Frontend** — React, MediaPipe JS, UI/UX
- **Backend/ML** — FastAPI, RandomForest model, data collection

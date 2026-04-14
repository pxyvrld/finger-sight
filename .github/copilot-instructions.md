# FingerSight — Copilot Instructions

## Project
Real-time Polish fingerspelling (PAP) recognition web app.
Stack: React 18 + MediaPipe (frontend), FastAPI + scikit-learn/Keras (backend).

## Code conventions
- Python: type hints, PEP8, docstrings on every function
- React: functional components, hooks only (no class components)
- All API responses: JSON with snake_case keys
- Landmark input is always List[List[float]] with shape (21, 3)
- Normalize landmarks relative to wrist (index 0) before model input

## File structure
fingersight/
  frontend/
    src/
      components/
      hooks/
      api/
  backend/
    app/
      main.py
      classifier.py
      normalizer.py
      models/
    data/
    train/

## Key classes
- HandClassifier: methods train(X, y), predict(landmarks) -> (str, float)
- LandmarkNormalizer: normalize(landmarks: list[list[float]]) -> list[float]

## API contract
POST /api/predict
  body: {"landmarks": [[x,y,z], ...]}  # 21 points
  response: {"letter": "A", "confidence": 0.97}

GET /api/ping
  response: {"status": "ok"}

GET /api/history
  response: {"history": ["A", "B", "M"]}

## Rules
- Do not use class components in React
- Do not hardcode model paths, use env vars or config file
- Always validate input in FastAPI endpoints with Pydantic
- Return HTTPException with detail string on errors
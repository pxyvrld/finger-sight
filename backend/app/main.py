"""FastAPI application — FingerSight backend."""

from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.classifier import HandClassifier
from app.config import CONFIDENCE_THRESHOLD, HISTORY_MAX_LENGTH, MODEL_PATH


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

classifier = HandClassifier()
history: deque[str] = deque(maxlen=HISTORY_MAX_LENGTH)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ML model on startup if the file exists."""
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        try:
            classifier.load(str(model_file))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Could not load model from {MODEL_PATH}: {exc}")
    else:
        print(
            f"[INFO] No model found at {MODEL_PATH}. "
            "Endpoint /api/predict will return null until model is trained."
        )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FingerSight API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /api/predict."""

    landmarks: list[list[float]]

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(
        cls, value: list[list[float]]
    ) -> list[list[float]]:
        """Ensure exactly 21 landmark points are provided."""
        if len(value) != 21:
            raise ValueError(
                f"landmarks must contain exactly 21 points, got {len(value)}"
            )
        for i, point in enumerate(value):
            if len(point) != 3:
                raise ValueError(
                    f"landmark[{i}] must have 3 coordinates (x, y, z), "
                    f"got {len(point)}"
                )
        return value


class PredictResponse(BaseModel):
    """Response body for POST /api/predict."""

    letter: str | None
    confidence: float


class PingResponse(BaseModel):
    """Response body for GET /api/ping."""

    status: str


class HistoryResponse(BaseModel):
    """Response body for GET /api/history."""

    history: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/ping", response_model=PingResponse)
async def ping() -> PingResponse:
    """Health-check endpoint.

    Returns:
        JSON with ``{"status": "ok"}``.
    """
    return PingResponse(status="ok")


@app.post("/api/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    """Classify a hand pose given 21 MediaPipe landmarks.

    Args:
        body: Request containing a list of 21 [x, y, z] landmark points.

    Returns:
        Predicted letter and confidence score.
        Returns ``{"letter": null, "confidence": 0.0}`` when confidence is
        below the threshold (``CONFIDENCE_THRESHOLD``).

    Raises:
        HTTPException 400: When the number of landmarks is not 21.
        HTTPException 500: On any internal error.
    """
    # Pydantic already validates landmark count; extra guard for clarity.
    if len(body.landmarks) != 21:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 21 landmarks, got {len(body.landmarks)}",
        )

    try:
        letter, confidence = classifier.predict(body.landmarks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if letter is not None:
        history.append(letter)

    return PredictResponse(letter=letter, confidence=confidence)


@app.get("/api/history", response_model=HistoryResponse)
async def get_history() -> HistoryResponse:
    """Return the list of recently recognised letters.

    Returns:
        JSON with ``{"history": [...]}``, newest entries last.
    """
    return HistoryResponse(history=list(history))

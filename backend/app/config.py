"""Configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


MODEL_PATH: str = os.getenv("MODEL_PATH", "app/models/model.pkl")
HISTORY_MAX_LENGTH: int = int(os.getenv("HISTORY_MAX_LENGTH", "50"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

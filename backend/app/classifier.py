"""Hand sign classifier wrapping a scikit-learn RandomForest model."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.normalizer import normalize


class HandClassifier:
    """Trains, saves, loads and runs inference for hand sign classification.

    The underlying model is a scikit-learn RandomForestClassifier.
    Landmarks are normalized before training/prediction.
    """

    def __init__(self) -> None:
        """Initialize with a default RandomForestClassifier."""
        self._model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self._trained: bool = False

    def train(self, X: list[list[float]], y: list[str]) -> None:
        """Fit the classifier on pre-normalized feature vectors.

        Args:
            X: List of 63-element feature vectors (already normalized).
            y: Corresponding letter labels.
        """
        self._model.fit(np.array(X), y)
        self._trained = True

    def predict(self, landmarks: list[list[float]]) -> tuple[str | None, float]:
        """Predict the letter shown by the hand pose.

        Normalizes raw MediaPipe landmarks, runs inference and returns the
        predicted letter with its confidence score.  Returns (None, 0.0) when
        no model is loaded or confidence is below the threshold defined in
        config.

        Args:
            landmarks: 21 [x, y, z] points from MediaPipe.

        Returns:
            Tuple of (predicted_letter_or_None, confidence).
        """
        if not self._trained:
            return None, 0.0

        from app.config import CONFIDENCE_THRESHOLD

        features = normalize(landmarks)
        X = np.array([features])
        proba = self._model.predict_proba(X)[0]
        confidence = float(proba.max())
        if confidence < CONFIDENCE_THRESHOLD:
            return None, 0.0
        letter: str = self._model.classes_[proba.argmax()]
        return letter, confidence

    def save(self, path: str) -> None:
        """Persist the trained model to disk as a pickle file.

        Args:
            path: Destination file path (e.g. 'app/models/model.pkl').
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: Source file path (e.g. 'app/models/model.pkl').

        Raises:
            FileNotFoundError: When the file does not exist.
        """
        with open(path, "rb") as fh:
            self._model = pickle.load(fh)
        self._trained = True

    def predict_features(
        self, features: list[list[float]]
    ) -> tuple[list[str], list[float]]:
        """Run batch inference on pre-normalized feature vectors.

        Used by training and evaluation scripts where landmarks have already
        been normalized and stored in landmarks.csv.

        Args:
            features: List of 63-element feature vectors (already normalized).

        Returns:
            Tuple of (list_of_predicted_labels, list_of_confidence_scores).
        """
        if not self._trained:
            return [], []

        X = np.array(features)
        proba = self._model.predict_proba(X)
        labels: list[str] = self._model.classes_[proba.argmax(axis=1)].tolist()
        confidences: list[float] = proba.max(axis=1).tolist()
        return labels, confidences

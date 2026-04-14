"""Training script — fits a RandomForest on landmarks.csv and saves the model.

Usage:
    cd backend
    python train/train.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Allow imports from backend root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.classifier import HandClassifier  # noqa: E402
from app.config import MODEL_PATH  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "landmarks.csv"


def load_data(csv_path: Path) -> tuple[list[list[float]], list[str]]:
    """Load and return features and labels from landmarks.csv.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple of (X, y) where X is a list of 63-element feature vectors and
        y is a list of label strings.
    """
    df = pd.read_csv(csv_path)
    y: list[str] = df["label"].tolist()
    X: list[list[float]] = df.drop(columns=["label"]).values.tolist()
    return X, y


def main() -> None:
    """Train the model and save it to MODEL_PATH."""
    if not DATA_PATH.exists():
        print(f"[ERROR] Training data not found: {DATA_PATH}")
        print("Run train/collect.py first to gather data.")
        sys.exit(1)

    print(f"Loading data from {DATA_PATH} …")
    X, y = load_data(DATA_PATH)
    print(f"  Samples: {len(X)}, Classes: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = HandClassifier()
    print("Training RandomForest …")
    clf.train(X_train, y_train)

    # Quick accuracy report on pre-normalized test features
    from sklearn.metrics import accuracy_score

    preds, _ = clf.predict_features(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Validation accuracy: {acc:.4f}")

    clf.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

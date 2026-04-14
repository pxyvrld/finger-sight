"""Evaluation script — prints accuracy and confusion matrix.

Usage:
    cd backend
    python train/evaluate.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.classifier import HandClassifier  # noqa: E402
from app.config import MODEL_PATH  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "landmarks.csv"


def main() -> None:
    """Evaluate the saved model and print a detailed report."""
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run train/train.py first.")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"[ERROR] Data not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    y: list[str] = df["label"].tolist()
    X: list[list[float]] = df.drop(columns=["label"]).values.tolist()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = HandClassifier()
    clf.load(MODEL_PATH)

    y_pred, _ = clf.predict_features(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {acc:.4f}")

    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    out = Path("confusion_matrix.png")
    plt.savefig(out)
    print(f"Confusion matrix saved to {out}")


if __name__ == "__main__":
    main()

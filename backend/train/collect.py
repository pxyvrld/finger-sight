"""Data collection script — captures hand landmarks from a webcam.

Usage:
    cd backend
    python train/collect.py --letter A [--samples 200] [--camera 0]

For each run the script appends rows to data/landmarks.csv.
Each row: label + 63 normalized landmark features (x0,y0,z0,…,x20,y20,z20).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalizer import normalize  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "landmarks.csv"

HEADER = ["label"] + [
    f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _write_row(label: str, features: list[float]) -> None:
    """Append one sample row to the landmarks CSV.

    Creates the file with a header row if it does not exist yet.

    Args:
        label: The letter this sample belongs to.
        features: 63 normalized feature values.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(HEADER)
        writer.writerow([label] + features)


def collect(letter: str, n_samples: int, camera_index: int) -> None:
    """Capture *n_samples* hand-pose samples for *letter* from the webcam.

    Displays a live preview with landmark overlay.  Press SPACE to start
    capturing; the script exits automatically after collecting enough samples.

    Args:
        letter: The letter label to assign to collected samples.
        n_samples: Number of valid samples to collect before exiting.
        camera_index: OpenCV camera device index.
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)

    collected = 0
    capturing = False

    print(f"Collecting {n_samples} samples for letter '{letter}'.")
    print("Press SPACE to start / stop.  Press Q to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while collected < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                    )
                    if capturing:
                        raw = [
                            [lm.x, lm.y, lm.z] for lm in hand_lm.landmark
                        ]
                        features = normalize(raw)
                        _write_row(letter, features)
                        collected += 1

            status = (
                f"CAPTURING ({collected}/{n_samples})"
                if capturing
                else "PAUSED — press SPACE"
            )
            cv2.putText(
                frame,
                f"Letter: {letter}  {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if capturing else (0, 0, 255),
                2,
            )
            cv2.imshow("FingerSight — Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capturing = not capturing
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {collected} samples → {CSV_PATH}")


def main() -> None:
    """Parse CLI arguments and start collection."""
    parser = argparse.ArgumentParser(description="Collect PAP landmark data.")
    parser.add_argument("--letter", required=True, help="Letter to collect (e.g. A)")
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of samples (default: 200)"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)"
    )
    args = parser.parse_args()
    collect(args.letter, args.samples, args.camera)


if __name__ == "__main__":
    main()

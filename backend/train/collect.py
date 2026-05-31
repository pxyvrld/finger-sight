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
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalizer import normalize  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "landmarks.csv"
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HEADER = ["label"] + [
    f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _ensure_model() -> None:
    """Download the hand landmarker .task model if not present."""
    if MODEL_PATH.exists():
        return
    print(f"Pobieranie modelu MediaPipe do {MODEL_PATH} …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Pobieranie zakonczone.")


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
    _ensure_model()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)

    collected = 0
    capturing = False

    print(f"Collecting {n_samples} samples for letter '{letter}'.")
    print("Press SPACE to start / stop.  Press Q to quit.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while collected < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    h, w = frame.shape[:2]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                    for pt in pts:
                        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

                    if capturing:
                        raw = [[lm.x, lm.y, lm.z] for lm in hand_lm]
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
    print(f"Done. Collected {collected} samples -> {CSV_PATH}")


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

"""Landmark normalization for hand pose classification."""

import math


def normalize(landmarks: list[list[float]]) -> list[float]:
    """Normalize hand landmarks relative to the wrist point.

    Translates all points so the wrist (index 0) is at the origin,
    then scales by the distance between the wrist and the base of the
    middle finger (index 9), and finally flattens to a 63-element list.

    Args:
        landmarks: List of 21 [x, y, z] points from MediaPipe.

    Returns:
        Flattened list of 63 normalized feature values.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    wrist = landmarks[0]
    wx, wy, wz = wrist[0], wrist[1], wrist[2]

    # Scale by wrist → middle finger MCP (index 9) distance
    ref = landmarks[9]
    scale = math.sqrt(
        (ref[0] - wx) ** 2 + (ref[1] - wy) ** 2 + (ref[2] - wz) ** 2
    )
    if scale == 0.0:
        scale = 1.0

    features: list[float] = []
    for point in landmarks:
        features.append((point[0] - wx) / scale)
        features.append((point[1] - wy) / scale)
        features.append((point[2] - wz) / scale)

    return features

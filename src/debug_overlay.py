"""Debug visualisation for offline state-detection reports."""

from __future__ import annotations

from pathlib import Path

import cv2

from src.state_detector import DetectionResult, ROI_BY_STATE


def save_debug_overlay(
    image_path: str | Path, result: DetectionResult, output_dir: str | Path
) -> Path:
    """Save an annotated screenshot showing the exact UI regions that matched."""
    source = Path(image_path)
    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image for debug overlay: {source}")

    height, width = frame.shape[:2]
    for index, (x1, y1, x2, y2) in enumerate(ROI_BY_STATE[result.state]):
        left, top = round(x1 * width), round(y1 * height)
        right, bottom = round(x2 * width), round(y2 * height)
        cv2.rectangle(frame, (left, top), (right, bottom), (30, 220, 30), 2)
        label = result.matched_features[index] if index < len(result.matched_features) else "matched_roi"
        cv2.putText(
            frame,
            label,
            (left, max(24, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )

    banner = f"{result.state}  confidence={result.confidence:.2%}"
    cv2.rectangle(frame, (12, height - 52), (min(width - 12, 520), height - 12), (0, 0, 0), -1)
    cv2.putText(
        frame,
        banner,
        (24, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{source.stem}_{result.state.lower()}_debug.png"
    if not cv2.imwrite(str(destination), frame):
        raise OSError(f"Could not write debug overlay: {destination}")
    return destination.resolve()

"""Strict offline inventory-window detector using live title, grid, and button evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config_loader import ROIConfig, ThresholdConfig, load_roi_config, load_thresholds_config, normalized_to_pixel_roi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_GET_TEMPLATE = PROJECT_ROOT / "assets" / "templates" / "live" / "get" / "frame_485_get_window.png"


def _load_image(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return frame
    return image


def _similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
    size = (180, 64)
    gray_a = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), size), (3, 3), 0)
    gray_b = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY), size), (3, 3), 0)
    template = (float(cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)[0, 0]) + 1.0) / 2.0
    edges_a = cv2.Canny(gray_a, 50, 150)
    edges_b = cv2.Canny(gray_b, 50, 150)
    overlap = np.count_nonzero((edges_a > 0) & (edges_b > 0)) / max(1, np.count_nonzero((edges_a > 0) | (edges_b > 0)))
    return float(np.clip(0.70 * template + 0.30 * overlap, 0.0, 1.0))


def _region(image: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bounds
    return image[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]


def detect_get_window(
    image: np.ndarray | str | Path,
    roi_config: ROIConfig | None = None,
    thresholds: ThresholdConfig | None = None,
) -> dict[str, Any]:
    """Require title, item-grid, and collect-button evidence from a live template."""
    frame = _load_image(image)
    config = roi_config or load_roi_config()
    active_thresholds = thresholds or load_thresholds_config()
    left, top, right, bottom = normalized_to_pixel_roi(config.rois["get_window"], frame.shape[1], frame.shape[0])
    crop = frame[top:bottom, left:right]
    template = cv2.imread(str(LIVE_GET_TEMPLATE), cv2.IMREAD_COLOR)
    if template is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "matched_features": [],
            "debug": {"roi_name": "get_window", "reason": "live_get_template_missing"},
        }
    regions = {
        "inventory_title": (0.0, 0.0, 0.70, 0.27),
        "item_grid": (0.0, 0.27, 1.0, 0.75),
        "collect_button": (0.0, 0.75, 1.0, 1.0),
    }
    scores = {name: _similarity(_region(crop, bounds), _region(template, bounds)) for name, bounds in regions.items()}
    # Three independent areas are used.  The title alone overlaps the task UI,
    # so at least two features must pass and the grid/button must be one of them.
    passes = {
        "inventory_title": scores["inventory_title"] >= 0.80,
        "item_grid": scores["item_grid"] >= 0.70,
        "collect_button": scores["collect_button"] >= 0.78,
    }
    detected = sum(passes.values()) >= 2 and (passes["item_grid"] or passes["collect_button"])
    confidence = float(np.mean(list(scores.values())))
    return {
        "detected": detected,
        "confidence": round(confidence, 4),
        "matched_features": [name for name, passed in passes.items() if passed],
        "debug": {
            "roi_name": "get_window",
            "feature_scores": {name: round(score, 4) for name, score in scores.items()},
            "detector_min_confidence": active_thresholds.min_confidence_for("GET"),
        },
    }

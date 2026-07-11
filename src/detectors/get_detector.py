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


def _find_panel(image: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Localize the dark inventory panel before scale-normalized comparison."""
    full_height, full_width = image.shape[:2]
    # The legacy ROI also contains the right-side quest list. Limit live panel
    # localization to the left portion so quest text cannot merge into the dark
    # inventory contour (the root cause in Pilot frame 440).
    search_width = round(full_width * 0.78)
    search = image[:, :search_width]
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    mask = (gray < 120).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = search.shape[:2]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, panel_width, panel_height = cv2.boundingRect(contour)
        if not (
            width * 0.25 <= panel_width <= width * 0.92
            and height * 0.20 <= panel_height <= height * 0.88
            and 0.9 <= panel_width / max(1, panel_height) <= 2.4
        ):
            continue
        rectangularity = cv2.contourArea(contour) / max(1, panel_width * panel_height)
        dark_ratio = float(np.mean(gray[y:y + panel_height, x:x + panel_width] < 120))
        if rectangularity < 0.45 or dark_ratio < 0.45:
            continue
        candidates.append((rectangularity * dark_ratio * panel_width * panel_height, (x, y, x + panel_width, y + panel_height)))
    if not candidates:
        return None, None
    _, bounds = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = bounds
    return search[y1:y2, x1:x2], bounds


def _panel_structure_scores(panel: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    """Score title/grid/button structure after panel localization."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    title = gray[:max(1, round(height * 0.27))]
    grid = gray[round(height * 0.27):round(height * 0.75)]
    button = gray[round(height * 0.75):]
    title_bright_ratio = float(np.mean(title >= 165))
    button_bright_ratio = float(np.mean(button >= 150))
    edges = cv2.Canny(grid, 35, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = 0
    for contour in contours:
        _, _, cell_width, cell_height = cv2.boundingRect(contour)
        if (
            width * 0.07 <= cell_width <= width * 0.24
            and height * 0.10 <= cell_height <= height * 0.28
            and 0.65 <= cell_width / max(1, cell_height) <= 1.8
        ):
            cells += 1
    scores = {
        "inventory_title": min(1.0, title_bright_ratio / 0.025),
        "item_grid": min(1.0, cells / 8.0),
        "collect_button": min(1.0, button_bright_ratio / 0.018),
    }
    return scores, {
        "title_bright_ratio": round(title_bright_ratio, 4),
        "grid_cell_candidates": cells,
        "button_bright_ratio": round(button_bright_ratio, 4),
    }


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
    localized_crop, panel_bbox = _find_panel(crop)
    # The canonical asset is already an ROI crop; its panel bounds are stable.
    template_height, template_width = template.shape[:2]
    template_panel_bbox = (
        round(template_width * 0.19),
        round(template_height * 0.17),
        round(template_width * 0.92),
        round(template_height * 0.84),
    )
    tx1, ty1, tx2, ty2 = template_panel_bbox
    localized_template = template[ty1:ty2, tx1:tx2]
    comparison_crop = localized_crop if localized_crop is not None else crop
    comparison_template = localized_template if localized_template is not None else template
    regions = {
        "inventory_title": (0.0, 0.0, 0.70, 0.27),
        "item_grid": (0.0, 0.27, 1.0, 0.75),
        "collect_button": (0.0, 0.75, 1.0, 1.0),
    }
    similarity_scores = {
        name: _similarity(
            _region(comparison_crop, bounds),
            _region(comparison_template, bounds),
        )
        for name, bounds in regions.items()
    }
    structure_scores, structure_debug = (
        _panel_structure_scores(localized_crop)
        if localized_crop is not None
        else ({name: 0.0 for name in regions}, {
            "title_bright_ratio": 0.0,
            "grid_cell_candidates": 0,
            "button_bright_ratio": 0.0,
        })
    )
    scores = {
        name: max(similarity_scores[name], structure_scores[name])
        for name in regions
    }
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
            "panel_bbox": list(panel_bbox) if panel_bbox is not None else None,
            "template_panel_bbox": list(template_panel_bbox) if template_panel_bbox is not None else None,
            "localized_panel_comparison": panel_bbox is not None and template_panel_bbox is not None,
            "similarity_scores": {name: round(score, 4) for name, score in similarity_scores.items()},
            "structure_scores": {name: round(score, 4) for name, score in structure_scores.items()},
            "structure_debug": structure_debug,
            "feature_scores": {name: round(score, 4) for name, score in scores.items()},
            "detector_min_confidence": active_thresholds.min_confidence_for("GET"),
        },
    }

"""Strict offline inventory-window detector using live title, grid, and button evidence."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from src.config_loader import ROIConfig, ThresholdConfig, load_roi_config, load_thresholds_config, normalized_to_pixel_roi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_GET_TEMPLATE = PROJECT_ROOT / "assets" / "templates" / "live" / "get" / "frame_485_get_window.png"
GET_CONFIG = PROJECT_ROOT / "config" / "fishing_v2.yaml"


@lru_cache(maxsize=1)
def _load_get_settings() -> dict[str, Any]:
    data = yaml.safe_load(GET_CONFIG.read_text(encoding="utf-8"))
    settings = data.get("get_detector") if isinstance(data, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"Missing get_detector config: {GET_CONFIG}")
    return settings


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


def _panel_geometry(
    bounds: tuple[int, int, int, int], width: int, height: int,
    geometry: Mapping[str, float],
) -> tuple[bool, dict[str, float]]:
    x1, y1, x2, y2 = bounds
    ratios = {
        "x1": x1 / max(1, width),
        "y1": y1 / max(1, height),
        "width": (x2 - x1) / max(1, width),
        "height": (y2 - y1) / max(1, height),
        "x2": x2 / max(1, width),
        "y2": y2 / max(1, height),
    }
    valid = bool(
        float(geometry["x1_min"]) <= ratios["x1"] <= float(geometry["x1_max"])
        and float(geometry["y1_min"]) <= ratios["y1"] <= float(geometry["y1_max"])
        and float(geometry["width_min"]) <= ratios["width"] <= float(geometry["width_max"])
        and float(geometry["height_min"]) <= ratios["height"] <= float(geometry["height_max"])
        and ratios["x2"] >= float(geometry["x2_min"])
        and float(geometry["y2_min"]) <= ratios["y2"] <= float(geometry["y2_max"])
    )
    return valid, {name: round(value, 4) for name, value in ratios.items()}


def _fixed_panel_bounds(
    width: int, height: int, fallback: Mapping[str, float]
) -> tuple[int, int, int, int]:
    """Return the fixed-UI panel bounds used only for dark-background fallback."""
    return (
        round(width * float(fallback["x1_ratio"])),
        round(height * float(fallback["y1_ratio"])),
        width,
        round(height * float(fallback["y2_ratio"])),
    )


def _find_panel(
    image: np.ndarray, localizer: Mapping[str, Any],
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, dict[str, Any]]:
    """Localize the dark inventory panel before scale-normalized comparison."""
    full_height, full_width = image.shape[:2]
    # The legacy ROI also contains the right-side quest list. Limit live panel
    # localization to the left portion so quest text cannot merge into the dark
    # inventory contour (the root cause in Pilot frame 440).
    search_width = round(full_width * float(localizer["search_width_ratio"]))
    search = image[:, :search_width]
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    mask = (gray < 120).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = search.shape[:2]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    candidate_debug: list[dict[str, Any]] = []
    for contour in contours:
        x, y, panel_width, panel_height = cv2.boundingRect(contour)
        basic_geometry = bool(
            width * 0.25 <= panel_width <= width * 0.92
            and height * 0.20 <= panel_height <= height * 0.88
            and 0.9 <= panel_width / max(1, panel_height) <= 2.4
        )
        if not basic_geometry:
            continue
        rectangularity = cv2.contourArea(contour) / max(1, panel_width * panel_height)
        dark_ratio = float(np.mean(gray[y:y + panel_height, x:x + panel_width] < 120))
        if rectangularity < 0.45 or dark_ratio < 0.45:
            continue
        bounds = (x, y, x + panel_width, y + panel_height)
        geometry_valid, geometry_ratios = _panel_geometry(
            bounds, width, height, localizer["geometry"]
        )
        candidate_debug.append({
            "bbox": list(bounds),
            "geometry_valid": geometry_valid,
            "geometry_ratios": geometry_ratios,
            "rectangularity": round(float(rectangularity), 4),
            "dark_ratio": round(dark_ratio, 4),
        })
        if geometry_valid:
            candidates.append((rectangularity * dark_ratio * panel_width * panel_height, bounds))
    if not candidates:
        return None, None, {
            "search_size": [width, height],
            "candidate_count": len(candidate_debug),
            "candidates": candidate_debug[:5],
        }
    _, bounds = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = bounds
    return search[y1:y2, x1:x2], bounds, {
        "search_size": [width, height],
        "candidate_count": len(candidate_debug),
        "candidates": candidate_debug[:5],
    }


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
    get_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require title, item-grid, and collect-button evidence from a live template."""
    frame = _load_image(image)
    config = roi_config or load_roi_config()
    active_thresholds = thresholds or load_thresholds_config()
    settings = get_settings or _load_get_settings()
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
    localized_crop, panel_bbox, localizer_debug = _find_panel(crop, settings["localizer"])
    localization_source = "geometry_valid_dark_contour" if localized_crop is not None else "none"
    fallback_debug: dict[str, Any] | None = None
    if localized_crop is None:
        search_width = round(crop.shape[1] * float(settings["localizer"]["search_width_ratio"]))
        fallback = settings["dark_scene_fallback"]
        fallback_bounds = _fixed_panel_bounds(search_width, crop.shape[0], fallback)
        fx1, fy1, fx2, fy2 = fallback_bounds
        fallback_crop = crop[fy1:fy2, fx1:fx2]
        fallback_scores, fallback_structure = _panel_structure_scores(fallback_crop)
        fallback_gray = cv2.cvtColor(fallback_crop, cv2.COLOR_BGR2GRAY)
        fallback_dark_ratio = float(np.mean(fallback_gray < 120))
        fallback_debug = {
            "bbox": list(fallback_bounds),
            "structure_scores": {name: round(score, 4) for name, score in fallback_scores.items()},
            "structure_debug": fallback_structure,
            "dark_ratio": round(fallback_dark_ratio, 4),
        }
        # A dark scene can merge the real panel with the background.  The
        # fixed-UI fallback is accepted only when the item grid itself is
        # strong; bright text/button-like pixels alone are never sufficient.
        if (
            fallback_structure["grid_cell_candidates"] >= int(fallback["min_grid_cells"])
            and fallback_structure["title_bright_ratio"] >= float(fallback["min_title_bright_ratio"])
            and fallback_structure["button_bright_ratio"] >= float(fallback["min_button_bright_ratio"])
            and fallback_dark_ratio >= float(fallback["min_dark_ratio"])
        ):
            localized_crop = fallback_crop
            panel_bbox = fallback_bounds
            localization_source = "fixed_geometry_strong_grid_fallback"
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
    detected = passes["item_grid"] and (passes["inventory_title"] or passes["collect_button"])
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
            "localization_source": localization_source,
            "localizer": localizer_debug,
            "fixed_fallback": fallback_debug,
            "similarity_scores": {name: round(score, 4) for name, score in similarity_scores.items()},
            "structure_scores": {name: round(score, 4) for name, score in structure_scores.items()},
            "structure_debug": structure_debug,
            "feature_scores": {name: round(score, 4) for name, score in scores.items()},
            "detector_min_confidence": active_thresholds.min_confidence_for("GET"),
        },
    }

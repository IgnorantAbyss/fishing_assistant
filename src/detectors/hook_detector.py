"""Offline detector for the fishing hook timing bar.

The detector uses only the configured ``hook_bar`` ROI in an image supplied by
the caller.  It does not capture the screen or send input events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config_loader import ROIConfig, ThresholdConfig, load_roi_config, load_thresholds_config, normalized_to_pixel_roi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "logs" / "hook_reports"


def _load_image(image: np.ndarray | str | Path) -> tuple[np.ndarray, Path | None]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return frame, path
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with three channels")
    return image.copy(), None


def _components(mask: np.ndarray, *, min_width: int = 8, min_height: int = 5) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [
        tuple(int(value) for value in stat)
        for stat in stats[1:count]
        if stat[2] >= min_width and stat[3] >= min_height and stat[4] >= min_width * min_height * 0.15
    ]


def _find_coloured_bar(hsv: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None, np.ndarray, np.ndarray]:
    """Find the lower-ROI red fill and adjacent cyan timing area."""
    height, width = hsv.shape[:2]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    red_mask = (((hue <= 10) | (hue >= 165)) & (saturation >= 80) & (value >= 80)).astype(np.uint8)
    cyan_mask = ((hue >= 88) & (hue <= 112) & (saturation >= 80) & (value >= 80)).astype(np.uint8)

    # The configured ROI includes the prompt above the bar.  Limiting candidates
    # to its lower part excludes unrelated red UI text in hook2.png.
    lower_y = int(height * 0.60)
    candidates: list[tuple[int, int, int, int]] = []
    for x, y, component_width, component_height, _ in _components(red_mask) + _components(cyan_mask):
        if y < lower_y or component_width < max(20, width // 30):
            continue
        candidates.append((x, y, x + component_width, y + component_height))
    if not candidates:
        return None, None, red_mask, cyan_mask

    # Merge only adjacent coloured pieces on a common row.  The actual bar has a
    # red fill directly next to a cyan area; separate player-name UI is not merged.
    candidates.sort(key=lambda box: box[0])
    groups: list[list[tuple[int, int, int, int]]] = []
    for box in candidates:
        if not groups:
            groups.append([box])
            continue
        previous = groups[-1][-1]
        overlaps_y = min(previous[3], box[3]) - max(previous[1], box[1]) >= 0
        if box[0] - previous[2] <= 24 and overlaps_y:
            groups[-1].append(box)
        else:
            groups.append([box])
    group = max(groups, key=lambda items: max(item[2] for item in items) - min(item[0] for item in items))
    bar = (
        min(item[0] for item in group),
        min(item[1] for item in group),
        max(item[2] for item in group),
        max(item[3] for item in group),
    )
    red_parts = [item for item in group if np.count_nonzero(red_mask[item[1]:item[3], item[0]:item[2]])]
    fill = max(red_parts, key=lambda item: (item[2] - item[0]) * (item[3] - item[1])) if red_parts else None
    return bar, fill, red_mask, cyan_mask


def _find_divider(hsv: np.ndarray, bar: tuple[int, int, int, int], fill: tuple[int, int, int, int] | None) -> int | None:
    left, top, right, bottom = bar
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white_mask = (saturation <= 70) & (value >= 180)
    window = white_mask[max(0, top - 8):min(hsv.shape[0], bottom + 8), left:right]
    required_height = max(6, int((bottom - top) * 0.55))
    columns = np.where(window.sum(axis=0) >= required_height)[0]
    if not len(columns):
        return None
    groups = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
    centres = [left + int(round(group.mean())) for group in groups if len(group)]
    if not centres:
        return None
    expected = fill[2] if fill is not None else (left + right) // 2
    return min(centres, key=lambda position: abs(position - expected))


def _save_debug_image(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    bar: tuple[int, int, int, int] | None,
    fill: tuple[int, int, int, int] | None,
    divider_x: int | None,
    source: Path | None,
) -> str | None:
    annotated = frame.copy()
    left, top, right, bottom = roi
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 215, 255), 2)
    cv2.putText(annotated, "hook_bar ROI", (left, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    if bar is not None:
        bx1, by1, bx2, by2 = bar
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 220, 0), 2)
    if fill is not None:
        fx1, fy1, fx2, fy2 = fill
        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
    if divider_x is not None and bar is not None:
        cv2.line(annotated, (divider_x, bar[1] - 5), (divider_x, bar[3] + 5), (255, 255, 255), 2)
    DEFAULT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stem = source.stem if source else "frame"
    destination = DEFAULT_DEBUG_DIR / f"{stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_hook.png"
    return str(destination.resolve()) if cv2.imwrite(str(destination), annotated) else None


def detect_hook_bar(
    image: np.ndarray | str | Path,
    roi_config: ROIConfig | None = None,
    thresholds: ThresholdConfig | None = None,
) -> dict[str, Any]:
    """Detect a coloured hook bar, fill ratio, and optional divider line.

    ``should_press_space`` is deliberately always false in this offline-only
    phase.  The measurement is provided for later validation, not automation.
    """
    frame, source = _load_image(image)
    active_roi = roi_config or load_roi_config()
    active_thresholds = thresholds or load_thresholds_config()
    roi = normalized_to_pixel_roi(active_roi.rois["hook_bar"], frame.shape[1], frame.shape[0])
    left, top, right, bottom = roi
    crop = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    local_bar, local_fill, red_mask, cyan_mask = _find_coloured_bar(hsv)
    if local_bar is None:
        debug_path = _save_debug_image(frame, roi, None, None, None, source)
        return {
            "detected": False,
            "confidence": 0.0,
            "bar_bbox": None,
            "fill_ratio": None,
            "divider_ratio": None,
            "perfect_zone_ratio": 0.95,
            "should_press_space": False,
            "matched_features": [],
            "debug": {"roi_name": "hook_bar", "raw_values": {"colour_pixels": 0}, "debug_image_path": debug_path},
        }

    bar = (left + local_bar[0], top + local_bar[1], left + local_bar[2], top + local_bar[3])
    fill = (
        (left + local_fill[0], top + local_fill[1], left + local_fill[2], top + local_fill[3])
        if local_fill is not None
        else None
    )
    local_divider = _find_divider(hsv, local_bar, local_fill)
    divider_x = left + local_divider if local_divider is not None else None
    bar_width = max(1, local_bar[2] - local_bar[0])
    fill_ratio = (local_fill[2] - local_bar[0]) / bar_width if local_fill is not None else 0.0
    divider_ratio = (local_divider - local_bar[0]) / bar_width if local_divider is not None else None
    features = ["hook_bar_rect"]
    if local_fill is not None:
        features.append("bar_fill")
    if local_divider is not None:
        features.append("divider_line")
    confidence = 0.60 + 0.20 * min(1.0, bar_width / max(1, crop.shape[1] * 0.40))
    confidence += 0.10 if local_fill is not None else 0.0
    confidence += 0.10 if local_divider is not None else 0.0
    confidence = round(min(1.0, confidence), 4)
    debug_path = _save_debug_image(frame, roi, bar, fill, divider_x, source)
    return {
        "detected": True,
        "confidence": confidence,
        "bar_bbox": list(bar),
        "fill_ratio": round(float(np.clip(fill_ratio, 0.0, 1.0)), 4),
        "divider_ratio": round(float(np.clip(divider_ratio, 0.0, 1.0)), 4) if divider_ratio is not None else None,
        "perfect_zone_ratio": 0.95,
        "should_press_space": False,
        "matched_features": features,
        "debug": {
            "roi_name": "hook_bar",
            "raw_values": {
                "bar_width_px": bar_width,
                "red_pixels": int(np.count_nonzero(red_mask)),
                "cyan_pixels": int(np.count_nonzero(cyan_mask)),
                "detector_min_confidence": active_thresholds.min_confidence_for("HOOK"),
            },
            "debug_image_path": debug_path,
        },
    }

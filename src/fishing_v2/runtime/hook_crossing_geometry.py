"""Action-specific Hook crossing geometry measured from the original frame.

Precise-ROI callers share the detector's bar-local geometry implementation.
Production reuses the detector result rather than invoking this compatibility
wrapper for a second measurement. Historical ``fill_ratio`` is not the cyan
progress endpoint relative to the fixed white divider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config_loader import ROIConfig, load_roi_config
from src.hook_bar_geometry import measure_bar_local_geometry


@dataclass(frozen=True)
class HookCrossingGeometry:
    divider_line_detected: bool = False
    divider_line_x: float | None = None
    divider_confidence: float = 0.0
    fill_endpoint_x: float | None = None

    def evidence(self) -> dict[str, Any]:
        return {
            "crossing_geometry_version": 1,
            "divider_line_detected": self.divider_line_detected,
            "divider_line_x": self.divider_line_x,
            "divider_confidence": self.divider_confidence,
            "fill_endpoint_x": self.fill_endpoint_x,
        }


def _load_frame(frame: Any) -> np.ndarray | None:
    if isinstance(frame, (str, Path)):
        return cv2.imread(str(frame), cv2.IMREAD_COLOR)
    if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    return None


def _contiguous_groups(columns: np.ndarray) -> list[np.ndarray]:
    if columns.size == 0:
        return []
    return list(np.split(columns, np.where(np.diff(columns) > 1)[0] + 1))


def measure_hook_crossing_geometry(
    frame: Any,
    roi_config: ROIConfig | None = None,
) -> HookCrossingGeometry:
    """Measure the fixed divider and cyan fill endpoint in ``hook_bar_precise``.

    The current red anchor supplies the vertical measurement band. A cyan
    segment must begin immediately right of its divider. Wide-ROI compatibility
    callers keep their original upper-band measurement.
    """
    image = _load_frame(frame)
    if image is None:
        return HookCrossingGeometry()
    active_roi = roi_config or load_roi_config()
    roi_name = "hook_bar_precise" if "hook_bar_precise" in active_roi.rois else "hook_bar"
    left, top, right, bottom = active_roi.pixel_roi(roi_name, image.shape[1], image.shape[0])
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return HookCrossingGeometry()

    if roi_name == "hook_bar_precise":
        geometry = measure_bar_local_geometry(
            cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
            canonical_band_height=max(16, round(image.shape[0] * 32 / 1440)),
        )
        return HookCrossingGeometry(
            divider_line_detected=geometry.divider_x is not None,
            divider_line_x=left + geometry.divider_x if geometry.divider_x is not None else None,
            divider_confidence=geometry.divider_confidence,
            fill_endpoint_x=left + geometry.fill_endpoint_x if geometry.fill_endpoint_x is not None else None,
        )

    # Legacy wide-ROI callers retain their historical measurement contract.
    band_height = min(crop.shape[0], max(16, round(image.shape[0] * 32 / 1440)))
    band = crop[:band_height]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    white = (saturation <= 60) & (value >= 180)
    minimum_column_height = max(6, round(band_height * 0.25))
    white_columns = np.where(white.sum(axis=0) >= minimum_column_height)[0]
    width = crop.shape[1]
    expected_divider = width * 0.58
    divider_window = max(10, round(width * 0.035))
    divider_columns = white_columns[
        (white_columns >= expected_divider - divider_window)
        & (white_columns <= expected_divider + divider_window)
    ]
    if divider_columns.size == 0:
        return HookCrossingGeometry()
    strengths = white[:, divider_columns].sum(axis=0)
    best_strength = strengths.max()
    strongest = divider_columns[strengths == best_strength]
    local_divider_x = float(min(strongest, key=lambda x: abs(float(x) - expected_divider)))
    divider_strength = float(best_strength)
    divider_confidence = min(1.0, divider_strength / max(1.0, band_height * 0.65))

    cyan = (hue >= 88) & (hue <= 112) & (saturation >= 100) & (value >= 120)
    cyan_columns = np.where(cyan.sum(axis=0) >= minimum_column_height)[0]
    cyan_groups = _contiguous_groups(cyan_columns)
    fill_group = next(
        (
            group for group in cyan_groups
            if group.size
            and local_divider_x + 0.5 <= float(group[0]) <= local_divider_x + 6.0
            and float(group[-1]) > local_divider_x
        ),
        None,
    )
    return HookCrossingGeometry(
        divider_line_detected=True,
        divider_line_x=left + local_divider_x,
        divider_confidence=round(divider_confidence, 4),
        fill_endpoint_x=(left + float(fill_group[-1])) if fill_group is not None else None,
    )

"""Same-frame bar-local divider/fill measurements shared by Hook consumers."""
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from src.hook_right_geometry import discover_right, anchor_pixels


class PassiveRightObservation(dict):
    """JSON-visible telemetry plus RAM-only packed identity (never serialized)."""
    identity: dict


def passive_right_observation(crop, hsv, geometry, roi, resolution):
    """Canonical owner, same crop/HSV/geometry; failure cannot alter crossing.

    Inclusive right pixel centres use screen coordinates in public telemetry;
    private identity uses native ROI coordinates like the offline prototype.
    No calls back into the detector or capture infrastructure.
    """
    started = perf_counter()
    observation = PassiveRightObservation(
        fillable_right_x=None, fillable_right_confidence=0.0,
        fillable_right_reason='unavailable', source='canonical_bar_local_structural',
        fill_endpoint_x=(roi[0]+geometry.fill_endpoint_x
                         if geometry.fill_endpoint_x is not None else None),
        distance_to_right=None,
    )
    observation.identity = {}
    try:
        edge = discover_right(crop, geometry, structural=True)
        identity = dict(roi_dimensions=[crop.shape[1],crop.shape[0]],
                        resolution=tuple(resolution), roi_bounds=tuple(roi),
                        anchor_bbox=geometry.anchor, divider_x=geometry.divider_x,
                        bar_local_y_top=geometry.anchor[1] if geometry.anchor else None,
                        bar_local_y_bottom=geometry.anchor[3] if geometry.anchor else None,
                        fill_endpoint_x=geometry.fill_endpoint_x,
                        fillable_right_x=edge['fillable_right_x'],
                        bar_right_confidence=edge['bar_right_confidence'])
        identity.update(anchor_pixels(hsv,geometry.anchor))
        observation.identity = identity
        right = edge['fillable_right_x']
        observation.update(
            fillable_right_x=roi[0]+right if right is not None else None,
            fillable_right_confidence=edge['bar_right_confidence'],
            fillable_right_reason=edge.get('structural_reason',edge['bar_right_reason']),
            distance_to_right=(right-geometry.fill_endpoint_x
                               if right is not None and geometry.fill_endpoint_x is not None else None),
            right_edge_support={key:edge.get(key) for key in (
                'edge_support_rows','total_rows','rail_endpoints','structural_corners')},
            geometry_identity={key:identity[key] for key in (
                'roi_dimensions','resolution','roi_bounds','anchor_bbox','divider_x',
                'bar_local_y_top','bar_local_y_bottom')},
        )
    except Exception as exc:
        observation.identity = {}
        observation.update(fillable_right_x=None, fillable_right_confidence=0.0,
                           fillable_right_reason=f'passive_error:{type(exc).__name__}')
    observation['measurement_cost_ms'] = (perf_counter()-started)*1000
    return observation


@dataclass(frozen=True)
class BarLocalGeometry:
    anchor: tuple[int, int, int, int] | None = None
    divider_band: tuple[int, int, int, int] | None = None
    fill_band: tuple[int, int, int, int] | None = None
    divider_x: float | None = None
    divider_confidence: float = 0.0
    fill_endpoint_x: float | None = None
    divider_candidates: tuple[int, ...] = ()
    reason: str = "bar_anchor_missing"

    def evidence(self, left: int, top: int) -> dict:
        def absolute(box):
            return [left + box[0], top + box[1], left + box[2], top + box[3]] if box else None
        return {
            "crossing_geometry_version": 1,
            "divider_line_detected": self.divider_x is not None,
            "divider_line_x": left + self.divider_x if self.divider_x is not None else None,
            "divider_confidence": self.divider_confidence,
            "fill_endpoint_x": left + self.fill_endpoint_x if self.fill_endpoint_x is not None else None,
            "bar_local_anchor_bbox": absolute(self.anchor),
            "divider_measurement_roi": absolute(self.divider_band),
            "fill_measurement_roi": absolute(self.fill_band),
            "bar_inner_left": left + self.anchor[0] if self.anchor else None,
            "bar_inner_top": top + self.anchor[1] if self.anchor else None,
            "bar_inner_bottom": top + self.anchor[3] if self.anchor else None,
            # Coloured support does not establish the right edge of dark UI.
            "bar_inner_right": None,
            "bar_right_reason": "dark_remaining_bar_edge_not_measured",
            "bar_local_geometry_reason": self.reason,
            "divider_candidates": [left + x for x in self.divider_candidates],
        }


def measure_bar_local_geometry(hsv: np.ndarray, anchor=None, *, canonical_band_height: int = 32) -> BarLocalGeometry:
    """Use the red frame as vertical support, never its interior hatch order.

    Pixel predicates and expected divider neighborhood are the existing
    canonical predicates. Only their spatial ownership changes: marker and
    background pixels above/below the current frame's red bar cannot vote.
    """
    height, width = hsv.shape[:2]
    hue, saturation, value = cv2.split(hsv)
    expected = width * 0.58
    radius = max(10, round(width * 0.035))
    if anchor is None:
        red = (((hue <= 10) | (hue >= 165)) & (saturation >= 80) & (value >= 80)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
        candidates = [(int(x), int(y), int(x+w), int(y+h)) for x,y,w,h,area in stats[1:]
                      if w >= max(200, int(width * 0.28)) and 10 <= h <= max(64, int(height * 0.68))
                      and abs(x+w-expected) <= radius]
        if len(candidates) != 1:
            return BarLocalGeometry(reason="bar_anchor_missing_or_conflicting")
        anchor = candidates[0]
    x1, y1, x2, y2 = anchor
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            and x2-x1 >= max(200, int(width * 0.28))
            and 10 <= y2-y1 <= max(64, int(height * 0.68))
            and abs(x2-expected) <= radius):
        return BarLocalGeometry(reason="bar_anchor_geometry_invalid")
    dx1, dx2 = max(x1, int(np.ceil(expected-radius))), min(width, int(np.floor(expected+radius))+1)
    divider_band = (dx1, y1, dx2, y2)
    white = (saturation[y1:y2, dx1:dx2] <= 60) & (value[y1:y2, dx1:dx2] >= 180)
    # Keep the pre-existing resolution-scaled pixel/ confidence denominators;
    # narrowing spatial support must not lower any acceptance threshold.
    canonical_band_height = min(height, canonical_band_height)
    required = max(6, round(canonical_band_height * 0.25))
    strengths = white.sum(axis=0)
    columns = np.flatnonzero(strengths >= required)
    candidate_columns = tuple(int(dx1+x) for x in columns)
    if not len(columns):
        return BarLocalGeometry(anchor, divider_band, divider_candidates=candidate_columns, reason="divider_line_missing")
    groups = np.split(columns, np.where(np.diff(columns) > 1)[0]+1)
    if len(groups) != 1:
        return BarLocalGeometry(anchor, divider_band, divider_candidates=candidate_columns, reason="divider_candidates_conflict")
    best = columns[strengths[columns] == strengths[columns].max()]
    divider = int(min(best, key=lambda x: abs(dx1+int(x)-expected))) + dx1
    confidence = min(1.0, float(strengths[divider-dx1]) / max(1.0, canonical_band_height*0.65))
    fill_band = (divider+1, y1, width, y2)
    cyan = ((hue[y1:y2, divider+1:] >= 88) & (hue[y1:y2, divider+1:] <= 112)
            & (saturation[y1:y2, divider+1:] >= 100) & (value[y1:y2, divider+1:] >= 120))
    columns = np.flatnonzero(cyan.sum(axis=0) >= required) + divider + 1
    groups = np.split(columns, np.where(np.diff(columns) > 1)[0]+1)
    group = next((g for g in groups if len(g) and divider+0.5 <= g[0] <= divider+6 and g[-1] > divider), None)
    return BarLocalGeometry(anchor, divider_band, fill_band, float(divider), round(confidence,4),
                            float(group[-1]) if group is not None else None, candidate_columns,
                            "bar_local_divider_and_fill" if group is not None else "bar_fill_missing")

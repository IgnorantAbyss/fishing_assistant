"""Offline WASD panel parser for static-purple and live-teal fishing UI glyphs."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config_loader import ROIConfig, ThresholdConfig, load_roi_config, load_thresholds_config, normalized_to_pixel_roi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_TEMPLATE_IMAGE = PROJECT_ROOT / "assets" / "reference" / "press.png"
LIVE_TEMPLATE_IMAGE = PROJECT_ROOT / "assets" / "templates" / "live" / "press" / "frame_472_press_sequence.png"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "logs" / "press_reports"
STATIC_BOOTSTRAP_SEQUENCE = "DSDSASSD"
LIVE_BOOTSTRAP_SEQUENCE = "ASDWWDWS"


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


def _letter_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for x, y, width, height, area in stats[1:count]:
        if y < int(mask.shape[0] * 0.45) or width < 8 or height < 14 or area < 70:
            continue
        candidates.append((int(x), int(y), int(x + width), int(y + height)))
    if not candidates:
        return []
    # Every key also has a lower directional arrow.  Live arrows can be tall
    # enough to satisfy the glyph dimensions, so keep only the top aligned row.
    top_row = min(box[1] for box in candidates)
    return sorted((box for box in candidates if box[1] <= top_row + 6), key=lambda box: box[0])


def _mixed_progress_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Split a connected eight-cell live progress row into glyph slots."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows = [
        tuple(int(value) for value in stat)
        for stat in stats[1:count]
        if stat[1] >= int(mask.shape[0] * 0.45)
        and stat[2] >= int(mask.shape[1] * 0.35)
        and stat[3] >= 35
    ]
    if not rows:
        return []
    x, y, width, height, _ = max(rows, key=lambda item: item[2] * item[3])
    glyph_height = max(18, min(36, round(height * 0.36)))
    return [
        (
            round(x + width * index / 8),
            y,
            round(x + width * (index + 1) / 8),
            min(mask.shape[0], y + glyph_height),
        )
        for index in range(8)
    ]


def _select_glyph_mask(crop: np.ndarray) -> tuple[str, np.ndarray, list[tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        "purple": ((hsv[:, :, 0] >= 125) & (hsv[:, :, 0] <= 170) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 65)).astype(np.uint8),
        "teal": ((hsv[:, :, 0] >= 75) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 65)).astype(np.uint8),
        # Live progress recolours completed/current/pending glyphs green, teal,
        # yellow, and red within the same row. A single-colour mask therefore
        # returned only the currently highlighted key (Pilot frame 428). This
        # mixed mask changes component extraction, not detector thresholds.
        "mixed_live": (
            (hsv[:, :, 1] >= 50)
            & (hsv[:, :, 2] >= 50)
        ).astype(np.uint8),
    }
    candidates = [
        (
            name,
            mask,
            _mixed_progress_boxes(mask) if name == "mixed_live" else _letter_boxes(mask),
        )
        for name, mask in masks.items()
    ]
    # Eight upper glyph components is a stronger signal than UI colour alone.
    return max(candidates, key=lambda item: (-(abs(len(item[2]) - 8)), len(item[2])))


def _glyph(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    tighten: bool = False,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    if tighten:
        rows, columns = np.where(crop > 0)
        if len(rows) and len(columns):
            crop = crop[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    return cv2.resize(crop, (24, 32), interpolation=cv2.INTER_NEAREST)


def _templates_from_crop(crop: np.ndarray, sequence: str) -> tuple[dict[str, list[np.ndarray]], str | None]:
    _, mask, boxes = _select_glyph_mask(crop)
    if len(boxes) != len(sequence):
        return {}, f"Expected {len(sequence)} key cells, found {len(boxes)}"
    templates: dict[str, list[np.ndarray]] = {}
    for label, box in zip(sequence, boxes, strict=True):
        templates.setdefault(label, []).append(_glyph(mask, box))
    return templates, None


@lru_cache(maxsize=1)
def _load_template_sets() -> tuple[dict[str, dict[str, list[np.ndarray]]], dict[str, str]]:
    """Load only two canonical ROI crops: static DSDSASSD and live ASDWWDWS."""
    template_sets: dict[str, dict[str, list[np.ndarray]]] = {}
    errors: dict[str, str] = {}
    static = cv2.imread(str(STATIC_TEMPLATE_IMAGE), cv2.IMREAD_COLOR)
    if static is not None:
        config = load_roi_config()
        left, top, right, bottom = normalized_to_pixel_roi(config.rois["press_sequence"], static.shape[1], static.shape[0])
        templates, error = _templates_from_crop(static[top:bottom, left:right], STATIC_BOOTSTRAP_SEQUENCE)
        if templates:
            template_sets["purple"] = templates
        elif error:
            errors["purple"] = error
    else:
        errors["purple"] = f"Missing static template image: {STATIC_TEMPLATE_IMAGE}"
    live = cv2.imread(str(LIVE_TEMPLATE_IMAGE), cv2.IMREAD_COLOR)
    if live is not None:
        templates, error = _templates_from_crop(live, LIVE_BOOTSTRAP_SEQUENCE)
        if templates:
            template_sets["teal"] = templates
        elif error:
            errors["teal"] = error
    else:
        errors["teal"] = f"Missing live template image: {LIVE_TEMPLATE_IMAGE}"
    return template_sets, errors


def _classify_candidates(
    glyph: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> tuple[str, float, list[dict[str, Any]]]:
    if not templates:
        return "?", 0.0, []
    scores = {
        label: max(float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0]) for template in variants)
        for label, variants in templates.items()
    }
    ranked = sorted(
        (
            {"key": key, "confidence": round(max(0.0, min(1.0, (score + 1.0) / 2.0)), 4)}
            for key, score in scores.items()
        ),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    return ranked[0]["key"], ranked[0]["confidence"], ranked[:3]


def _horizontal_groups(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if not len(indices):
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value > previous + 1:
            groups.append((start, previous + 1))
            start = value
        previous = value
    groups.append((start, previous + 1))
    return groups


def _measure_panel_geometry(
    gray: np.ndarray,
    edges: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    baseline_hint: bool,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    panel_gray = gray[y1:y2, x1:x2]
    panel_edges = edges[y1:y2, x1:x2]
    panel_h, panel_w = panel_gray.shape[:2]
    dark_ratio = float(np.mean(panel_gray < 115)) if panel_gray.size else 0.0
    strong_columns = np.sum(panel_edges > 0, axis=0) >= max(8, round(panel_h * 0.45))
    divider_groups = _horizontal_groups(strong_columns)
    divider_centres = [round((start + end - 1) / 2) for start, end in divider_groups]
    divider_centres = [value for value in divider_centres if 2 <= value <= panel_w - 3]
    divider_count = len(divider_centres)
    estimated_slots = max(4, min(14, round(panel_w / max(1.0, panel_h * 0.64))))
    grid_match_count = sum(
        any(abs(centre - expected) <= 5 for centre in divider_centres)
        for expected in (round(index * panel_w / estimated_slots) for index in range(1, estimated_slots))
    )
    aspect = panel_w / max(1, panel_h)
    geometry_score = max(0.0, min(1.0, 1.0 - abs(aspect - 6.4) / 3.0))
    dark_score = max(0.0, min(1.0, (dark_ratio - 0.25) / 0.45))
    divider_score = max(0.0, min(1.0, grid_match_count / max(1, estimated_slots - 1)))
    panel_confidence = 0.45 + 0.25 * geometry_score + 0.20 * dark_score + 0.10 * divider_score
    panel_candidate = bool(baseline_hint or dark_ratio >= 0.20 or grid_match_count >= 2)
    panel_present = bool(
        4.5 <= aspect <= 10.0
        and dark_ratio >= 0.38
        and grid_match_count >= max(4, round((estimated_slots - 1) * 0.75))
    )
    reason = (
        "structural_panel_present" if panel_present
        else "panel_geometry_incomplete" if panel_candidate
        else "panel_not_found"
    )
    boundaries = [round(index * panel_w / estimated_slots) for index in range(estimated_slots + 1)]
    slot_boxes = [
        (x1 + boundaries[index], y1, x1 + boundaries[index + 1], y2)
        for index in range(estimated_slots)
    ]
    return {
        "panel_candidate": panel_candidate,
        "panel_present": panel_present,
        "reason": reason,
        "panel_bbox": bbox,
        "dark_ratio": dark_ratio,
        "divider_count": divider_count,
        "grid_match_count": grid_match_count,
        "slot_count": estimated_slots,
        "geometry_score": geometry_score,
        "panel_confidence": max(0.0, min(1.0, panel_confidence)) if panel_candidate else 0.0,
        "slot_boxes": slot_boxes,
        "baseline_hint": baseline_hint,
    }


def _find_panel_geometry(crop: np.ndarray) -> dict[str, Any]:
    """Find the PRESS grid by timer hints plus repeated cell geometry."""
    height, width = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    yellow = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] >= 75)
        & (hsv[:, :, 2] >= 70)
    ).astype(np.uint8) * 255
    yellow[: int(height * 0.55)] = 0
    yellow = cv2.morphologyEx(
        yellow,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)),
    )
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    baselines: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        if (
            candidate_width >= int(width * 0.30)
            and candidate_height <= 18
            and int(height * 0.55) <= y <= height - 1
        ):
            baselines.append((x, y, candidate_width, candidate_height))
    panel_width = round(width * 0.617)
    panel_height = round(height * 0.298)
    candidates: dict[tuple[int, int, int, int], bool] = {}
    for x, y, candidate_width, _ in baselines:
        if candidate_width < width * 0.90:
            x1 = max(0, min(width - panel_width, x))
            y2 = max(panel_height, min(height, y + 1))
            candidates[(x1, y2 - panel_height, x1 + panel_width, y2)] = True
    for x_ratio in (0.176, 0.190, 0.204):
        for y_ratio in (0.49, 0.54, 0.59, 0.657):
            x1 = max(0, min(width - panel_width, round(width * x_ratio)))
            y1 = max(0, min(height - panel_height, round(height * y_ratio)))
            candidates.setdefault((x1, y1, x1 + panel_width, y1 + panel_height), False)
    measurements = [
        _measure_panel_geometry(gray, edges, bbox, baseline_hint=hint)
        for bbox, hint in candidates.items()
    ]
    return max(
        measurements,
        key=lambda item: (
            item["panel_present"],
            item["grid_match_count"],
            item["dark_ratio"],
            item["baseline_hint"],
        ),
    )


def _decode_panel_slots(
    crop: np.ndarray,
    geometry: dict[str, Any],
    templates: dict[str, list[np.ndarray]],
) -> list[dict[str, Any]]:
    if not geometry["panel_present"]:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    key_boxes: list[dict[str, Any]] = []
    occupied_started = False
    for x1, y1, x2, y2 in geometry["slot_boxes"]:
        cell = hsv[y1:y2, x1:x2]
        if not cell.size:
            continue
        # Letter glyphs occupy the upper half; directional arrows below them
        # are deliberately excluded from classification.
        glyph_bottom = max(1, round(cell.shape[0] * 0.56))
        glyph_region = cell[2:glyph_bottom, 3:max(4, cell.shape[1] - 3)]
        coloured = (
            (glyph_region[:, :, 1] >= 50)
            & (glyph_region[:, :, 2] >= 45)
        )
        occupied = float(np.mean(coloured)) >= 0.035
        if not occupied:
            if occupied_started:
                break
            continue
        occupied_started = True
        mask = coloured.astype(np.uint8)
        key, confidence, top_candidates = _classify_candidates(
            _glyph(mask, (0, 0, mask.shape[1], mask.shape[0]), tighten=True),
            templates,
        )
        key_boxes.append({
            "key": key,
            "bbox": [x1, y1, x2, y1 + glyph_bottom],
            "confidence": round(confidence, 4),
            "top_candidates": top_candidates,
        })
    return key_boxes


ARROW_TO_KEY = {"LEFT": "A", "DOWN": "S", "RIGHT": "D", "UP": "W"}
PRESS_PROGRESS_HUE_SPREAD_MIN = 22.0


def _component_records(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return bounded structural components; full-width panel/background is noise."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    height, width = mask.shape[:2]
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        centroid_x, centroid_y = (float(value) for value in centroids[label])
        reasons: list[str] = []
        if area < 8:
            reasons.append("area_too_small")
        if component_width < 2 or component_height < 3:
            reasons.append("bbox_too_small")
        if component_width >= width * 0.85:
            reasons.append("background_spans_slot_width")
        if centroid_x < width * 0.08 or centroid_x > width * 0.92:
            reasons.append("centroid_outside_slot_centre_band")
        components.append({
            "label": label,
            "bbox": [x, y, x + component_width, y + component_height],
            "area": area,
            "centroid": [round(centroid_x, 3), round(centroid_y, 3)],
            "structural": not reasons,
            "rejection_reasons": reasons,
        })
    return mask.astype(np.uint8), components


def _split_letter_arrow_regions(
    colour_mask: np.ndarray,
    arrow_colour_mask: np.ndarray,
) -> tuple[int, int, dict[str, Any]]:
    """Jointly assign letter/arrow components across the complete inner slot.

    The panel moves vertically during animation, so no role is tied to a fixed
    percentage of slot height.  Wide connected background bands are retained
    for diagnostics but cannot establish either glyph role.
    """
    height, width = colour_mask.shape[:2]
    _, broad_components = _component_records(colour_mask)
    _, arrow_components = _component_records(arrow_colour_mask)
    letter_candidates = [
        {**item, "component_source": source}
        for source, components in (
            ("broad_colour", broad_components),
            ("arrow_colour", arrow_components),
        )
        for item in components
        if item["structural"] and int(item["bbox"][1]) < height * 0.78
    ]
    paired_arrow_candidates: list[dict[str, Any]] = []
    component_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for letter_candidate in letter_candidates:
        letter = letter_candidate
        letter_bottom = int(letter["bbox"][3])
        for item in arrow_components:
            arrow_y = float(item["centroid"][1])
            arrow_width = int(item["bbox"][2]) - int(item["bbox"][0])
            arrow_height = int(item["bbox"][3]) - int(item["bbox"][1])
            reasons = list(item["rejection_reasons"])
            if int(item["area"]) < 25:
                reasons.append("area_too_small_for_joint_arrow_role")
            if arrow_y <= letter_bottom - 1:
                reasons.append("not_below_selected_letter")
            if (
                letter.get("component_source") == "arrow_colour"
                and int(letter["label"]) == int(item["label"])
            ):
                reasons.append("same_component_as_letter")
            if int(item["area"]) > 180:
                reasons.append("area_too_large_for_arrow")
            if arrow_width > width * 0.58 or arrow_height > height * 0.36:
                reasons.append("bbox_too_large_for_arrow")
            centre_quality = max(
                0.0,
                1.0 - abs(float(item["centroid"][0]) - width * 0.5)
                / max(1.0, width * 0.5),
            )
            vertical_gap = max(0.0, float(item["bbox"][1]) - letter_bottom)
            proximity_quality = max(0.0, 1.0 - vertical_gap / max(1.0, height * 0.32))
            candidate = {
                **item,
                "paired_letter_label": int(letter["label"]),
                "paired_letter_source": str(letter["component_source"]),
                "eligible_for_arrow_role": not reasons,
                "arrow_role_rejection_reasons": reasons,
                "arrow_role_score": round(
                    0.55 * centre_quality
                    + 0.25 * proximity_quality
                    + 0.20 * min(1.0, int(item["area"]) / 55.0),
                    4,
                ),
            }
            paired_arrow_candidates.append(candidate)
            if candidate["eligible_for_arrow_role"]:
                letter_area_quality = min(1.0, int(letter["area"]) / 140.0)
                pair_score = (
                    float(candidate["arrow_role_score"])
                    + 0.35 * letter_area_quality
                )
                component_pairs.append((pair_score, letter, candidate))
    selected_pair = max(component_pairs, key=lambda item: item[0], default=None)
    if selected_pair is not None:
        _, letter, selected_arrow = selected_pair
    else:
        # Preserve the previous low-saturation fallback: the strongest
        # arrow-colour structure is treated as the upper glyph boundary.
        fallback_letters = [
            item for item in letter_candidates
            if item["component_source"] == "arrow_colour"
        ]
        letter = max(
            fallback_letters,
            key=lambda item: (
                min(int(item["area"]), 320),
                -abs(float(item["centroid"][0]) - width * 0.5),
            ),
            default=None,
        )
        selected_arrow = None
    if selected_arrow is not None:
        arrow_start = max(
            int(letter["bbox"][3]),
            int(selected_arrow["bbox"][1]) - 2,
        )
        arrow_end = min(height, int(selected_arrow["bbox"][3]) + 2)
        assignment_mode = "joint_full_slot_components"
    else:
        # Compatibility for low-saturation frames where letter/arrow pixels
        # merge in the broad mask.  This fallback is dynamic below the selected
        # structure and no longer imposes a 40% lower bound.
        arrow_start = (
            max(0, int(letter["bbox"][3]) + 1)
            if letter is not None else max(0, round(height * 0.22))
        )
        arrow_start = min(height - 1, arrow_start)
        arrow_end = max(arrow_start + 1, min(height, round(height * 0.98)))
        assignment_mode = "dynamic_below_letter_fallback"
    return arrow_start, arrow_end, {
        "letter_component_candidates": letter_candidates,
        "broad_colour_component_candidates": broad_components,
        "selected_letter_component": letter,
        "joint_arrow_component_candidates": paired_arrow_candidates,
        "selected_joint_arrow_component": selected_arrow,
        "component_assignment_mode": assignment_mode,
        "letter_region": [0, 0, width, arrow_start],
        "arrow_region": [0, arrow_start, width, arrow_end],
        "regions_overlap": False,
    }


def _extract_arrow_glyph(
    mask: np.ndarray,
) -> tuple[np.ndarray | None, list[int] | None, list[dict[str, Any]], str]:
    """Select a lower-centred arrow-shaped component, never merely the largest blob."""
    raw_mask = mask.astype(np.uint8)
    working_mask = raw_mask
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        working_mask, connectivity=8
    )
    height, width = mask.shape[:2]
    candidates: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        centroid_x, centroid_y = (float(value) for value in centroids[label])
        rejection_reasons: list[str] = []
        if area < 6 or area > 240:
            rejection_reasons.append("area_out_of_range")
        if component_width < 3 or component_height < 3:
            rejection_reasons.append("bbox_too_small")
        if component_width >= width * 0.72:
            rejection_reasons.append("bbox_too_wide_for_arrow")
        if centroid_y < height * 0.08 or centroid_y > height * 0.72:
            rejection_reasons.append("centroid_outside_expected_lower_slot_band")
        area_quality = min(1.0, area / 55.0)
        size_quality = min(1.0, component_width / 9.0, component_height / 7.0)
        centre_quality = max(0.0, 1.0 - abs(centroid_x - (width - 1) * 0.5) / max(1.0, width * 0.5))
        vertical_quality = max(
            0.0,
            1.0 - abs(centroid_y - height * 0.32) / max(1.0, height * 0.32),
        )
        selection_score = (
            0.35 * area_quality
            + 0.25 * size_quality
            + 0.25 * centre_quality
            + 0.15 * vertical_quality
        )
        candidates.append({
            "label": label,
            "bbox": [x, y, x + component_width, y + component_height],
            "area": area,
            "centroid": [round(centroid_x, 3), round(centroid_y, 3)],
            "selection_score": round(selection_score, 4),
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        })
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        for item in candidates:
            item["selected"] = False
        return None, None, candidates, "no_eligible_arrow_component"
    selected = max(eligible, key=lambda item: (item["selection_score"], item["area"]))
    for item in candidates:
        item["selected"] = item is selected
    x, y, x2, y2 = (int(value) for value in selected["bbox"])
    component = (labels[y:y2, x:x2] == int(selected["label"])).astype(np.uint8)
    return component, [x, y, x2, y2], candidates, "best_area_size_centre_position_score"


def _normalised_projection_slope(values: np.ndarray) -> float:
    if len(values) < 2 or float(np.max(values)) <= 0.0:
        return 0.0
    axis = np.linspace(-1.0, 1.0, len(values))
    normalised = values.astype(np.float64) / float(np.max(values))
    slope = float(np.dot(axis, normalised - np.mean(normalised)) / np.dot(axis, axis))
    return max(-1.0, min(1.0, slope))


def _classify_arrow(mask: np.ndarray) -> dict[str, Any]:
    component, bbox, component_candidates, selection_reason = _extract_arrow_glyph(mask)
    if component is None or bbox is None:
        return {
            "arrow_direction": None,
            "arrow_confidence": 0.0,
            "second_direction": None,
            "ambiguity_margin": 0.0,
            "mapped_key": None,
            "arrow_bbox": None,
            "arrow_top_candidates": [],
            "arrow_component_candidates": component_candidates,
            "arrow_component_selection_reason": selection_reason,
            "arrow_geometry_features": {},
            "arrow_geometry_scores": {},
        }
    component_bool = component.astype(bool)
    height, width = component_bool.shape[:2]
    total = max(1.0, float(np.sum(component)))
    rows, columns = np.where(component_bool)
    moments = cv2.moments(component.astype(np.uint8), binaryImage=True)
    centroid_x = float(
        (moments["m10"] / max(1.0, moments["m00"])) / max(1, width - 1)
    )
    centroid_y = float(
        (moments["m01"] / max(1.0, moments["m00"])) / max(1, height - 1)
    )
    left = float(np.sum(component_bool[:, : max(1, width // 2)]))
    right = total - left
    top = float(np.sum(component_bool[: max(1, height // 2), :]))
    bottom = total - top
    column_slope = _normalised_projection_slope(np.sum(component_bool, axis=0))
    row_slope = _normalised_projection_slope(np.sum(component_bool, axis=1))
    quarter_width = max(1, int(np.ceil(width * 0.25)))
    quarter_height = max(1, int(np.ceil(height * 0.25)))

    def _orthogonal_span(values: np.ndarray, size: int) -> float:
        return float(np.ptp(values) + 1) / size if len(values) else 0.0

    left_span = _orthogonal_span(rows[columns < quarter_width], height)
    right_span = _orthogonal_span(rows[columns >= width - quarter_width], height)
    top_span = _orthogonal_span(columns[rows < quarter_height], width)
    bottom_span = _orthogonal_span(columns[rows >= height - quarter_height], width)
    features = {
        "centroid_horizontal": 2.0 * (0.5 - centroid_x),
        "centroid_vertical": 2.0 * (0.5 - centroid_y),
        "mass_horizontal": (left - right) / total,
        "mass_vertical": (top - bottom) / total,
        "projection_horizontal": -column_slope,
        "projection_vertical": -row_slope,
        "base_span_horizontal": left_span - right_span,
        "base_span_vertical": top_span - bottom_span,
    }
    horizontal = (
        0.45 * features["centroid_horizontal"]
        + 0.30 * features["mass_horizontal"]
        + 0.25 * features["projection_horizontal"]
    )
    vertical = (
        0.45 * features["centroid_vertical"]
        + 0.30 * features["mass_vertical"]
        + 0.25 * features["projection_vertical"]
    )
    axis_bias = (height - width) / max(1.0, height + width)
    covariance = np.cov(np.vstack((columns, rows)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    largest_eigenvalue = float(np.max(eigenvalues))
    pca_anisotropy = (
        (largest_eigenvalue - float(np.min(eigenvalues))) / largest_eigenvalue
        if largest_eigenvalue > 0.0 else 0.0
    )
    # A triangle's longest mass axis follows its base, perpendicular to the
    # pointing direction.  Positive values therefore favour LEFT/RIGHT.
    pca_axis_bias = (
        abs(float(principal[1])) - abs(float(principal[0]))
    ) * pca_anisotropy
    horizontal += 0.35 * features["base_span_horizontal"]
    vertical += 0.35 * features["base_span_vertical"]
    geometry_scores = {
        "RIGHT": horizontal,
        "LEFT": -horizontal,
        "DOWN": vertical,
        "UP": -vertical,
    }
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_base_direction = None
    contour_base_quality = 0.0
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2)
        edge_candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for index, start in enumerate(hull):
            end = hull[(index + 1) % len(hull)]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length <= 0.0:
                continue
            horizontal_edge = abs(int(vector[0])) >= abs(int(vector[1]))
            alignment = max(abs(int(vector[0])), abs(int(vector[1]))) / length
            midpoint = (start + end) * 0.5
            if horizontal_edge:
                boundary_distance = min(midpoint[1], height - 1 - midpoint[1]) / max(1.0, (height - 1) * 0.5)
            else:
                boundary_distance = min(midpoint[0], width - 1 - midpoint[0]) / max(1.0, (width - 1) * 0.5)
            edge_score = length * alignment * (1.0 - 0.70 * boundary_distance)
            edge_candidates.append((edge_score, start, end))
        if edge_candidates:
            edge_score, base_start, base_end = max(edge_candidates, key=lambda item: item[0])
            base_vector = base_end - base_start
            relative = hull - base_start
            distances = (
                base_vector[0] * relative[:, 1] - base_vector[1] * relative[:, 0]
            ) / max(1.0, float(np.linalg.norm(base_vector)))
            tip = hull[int(np.argmax(np.abs(distances)))]
            direction_vector = tip - (base_start + base_end) * 0.5
            if abs(float(direction_vector[0])) > abs(float(direction_vector[1])):
                contour_base_direction = "RIGHT" if direction_vector[0] > 0 else "LEFT"
            else:
                contour_base_direction = "DOWN" if direction_vector[1] > 0 else "UP"
            contour_base_quality = float(
                edge_score / max(1.0, float(max(width, height)))
            )
            if contour_base_quality >= 0.55:
                geometry_scores[contour_base_direction] += 0.50
    ranked = sorted(geometry_scores.items(), key=lambda item: item[1], reverse=True)
    direction, best = ranked[0]
    second_direction, second = ranked[1]
    signal_margin = max(0.0, best - second)
    confidence = max(0.0, min(1.0, 0.55 + 1.35 * max(0.0, best) + 1.10 * signal_margin))
    classified = bool(best >= 0.055 and signal_margin >= 0.025)
    return {
        "arrow_direction": direction if classified else None,
        "arrow_confidence": round(confidence if classified else 0.0, 4),
        "second_direction": second_direction,
        "ambiguity_margin": round(signal_margin, 4),
        "mapped_key": ARROW_TO_KEY.get(direction) if classified else None,
        "arrow_bbox": bbox,
        "arrow_top_candidates": [
            {
                "key": ARROW_TO_KEY[candidate],
                "direction": candidate,
                "confidence": round(max(0.0, min(1.0, 0.5 + score)), 4),
            }
            for candidate, score in ranked[:2]
        ],
        "arrow_component_candidates": component_candidates,
        "arrow_component_selection_reason": selection_reason,
        "arrow_geometry_features": {
            **{key: round(value, 4) for key, value in features.items()},
            "axis_aspect_bias": round(axis_bias, 4),
            "pca_axis_bias": round(pca_axis_bias, 4),
            "pca_anisotropy": round(pca_anisotropy, 4),
            "contour_base_direction": contour_base_direction,
            "contour_base_quality": round(contour_base_quality, 4),
        },
        "arrow_geometry_scores": {key: round(value, 4) for key, value in geometry_scores.items()},
    }


def _decode_arrow_slots(crop: np.ndarray, geometry: dict[str, Any]) -> dict[str, Any]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    slots: list[dict[str, Any]] = []
    for index, (x1, y1, x2, y2) in enumerate(geometry["slot_boxes"]):
        slot_height = y2 - y1
        extended_y2 = min(hsv.shape[0], y2 + max(4, round(slot_height * 0.20)))
        cell = hsv[y1:extended_y2, x1:x2]
        if not cell.size:
            continue
        height, width = cell.shape[:2]
        colour = (cell[:, :, 1] >= 60) & (cell[:, :, 2] >= 60)
        arrow_colour = (cell[:, :, 1] >= 75) & (cell[:, :, 2] >= 70)
        inner_colour = colour[:, 3:max(4, width - 3)]
        inner_arrow_colour = arrow_colour[:, 3:max(4, width - 3)]
        arrow_start, arrow_end, region_debug = _split_letter_arrow_regions(
            inner_colour,
            inner_arrow_colour,
        )
        arrow_value = cell[arrow_start:arrow_end, 3:max(4, width - 3), 2]
        value_range = int(np.max(arrow_value)) - int(np.min(arrow_value))
        if arrow_value.size and value_range >= 25:
            arrow_value_threshold, arrow_shape = cv2.threshold(
                arrow_value,
                0,
                1,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            arrow_mask = arrow_shape.astype(bool)
            arrow_shape_segmentation = "local_value_otsu"
        else:
            arrow_value_threshold = 0.0
            arrow_mask = inner_arrow_colour[arrow_start:arrow_end]
            arrow_shape_segmentation = "fixed_colour_mask"
        arrow = _classify_arrow(arrow_mask)
        for candidate in arrow["arrow_component_candidates"]:
            centroid = candidate.get("centroid")
            if isinstance(centroid, list) and len(centroid) == 2:
                slot_centroid_y = float(centroid[1]) + arrow_start
                candidate["slot_centroid"] = [
                    round(float(centroid[0]) + 3.0, 3),
                    round(slot_centroid_y, 3),
                ]
                candidate["centroid_in_slot_lower_half"] = bool(
                    slot_centroid_y >= height * 0.5
                )
        selected_component = next(
            (
                item for item in arrow["arrow_component_candidates"]
                if item["eligible"] and item["bbox"] == arrow["arrow_bbox"]
            ),
            None,
        )
        selected_letter = region_debug.get("selected_letter_component")
        letter_pixels = int(selected_letter["area"]) if selected_letter else 0
        arrow_pixels = int(selected_component["area"]) if selected_component else 0
        raw_coloured_pixels = int(np.sum(inner_colour))
        structural_pixels = letter_pixels + arrow_pixels
        combined_effect_candidates = [
            item for item in region_debug.get(
                "broad_colour_component_candidates", ()
            )
            if (
                "background_spans_slot_width" in item.get("rejection_reasons", ())
                and int(item["bbox"][3]) - int(item["bbox"][1]) >= height * 0.45
            )
        ]
        combined_effect_component = max(
            combined_effect_candidates,
            key=lambda item: item["area"],
            default=None,
        )
        valid_letter_arrow_pair = bool(
            selected_letter is not None and selected_component is not None
        )
        if valid_letter_arrow_pair:
            occupancy = "OCCUPIED"
            occupancy_confidence = min(1.0, 0.55 + structural_pixels / 100.0)
            occupancy_reason = "valid_letter_arrow_pair"
        elif (
            selected_letter is not None
            or selected_component is not None
            or any(
                item.get("structural") is True
                for item in region_debug.get(
                    "letter_component_candidates", ()
                )
            )
            or any(
                item.get("eligible") is True
                for item in arrow.get("arrow_component_candidates", ())
            )
        ):
            occupancy = "UNCERTAIN"
            occupancy_confidence = 0.5
            occupancy_reason = "unpaired_structural_component"
        else:
            occupancy = "EMPTY"
            occupancy_confidence = 1.0
            occupancy_reason = "no_glyph_structure"
        coloured_hues = cell[:, :, 0][colour]
        median_hue = float(np.median(coloured_hues)) if len(coloured_hues) else None
        slot = {
            "index": index,
            "bbox": [x1, y1, x2, y2],
            "recognition_bbox": [x1, y1, x2, extended_y2],
            "occupancy": occupancy,
            "occupancy_confidence": round(float(occupancy_confidence), 4),
            "occupancy_reason": occupancy_reason,
            "coloured_pixel_count": raw_coloured_pixels,
            "raw_coloured_pixel_count": raw_coloured_pixels,
            "structural_pixel_count": structural_pixels,
            "combined_input_effect_component": combined_effect_component,
            "letter_pixel_count": letter_pixels,
            "arrow_pixel_count": arrow_pixels,
            "arrow_shape_segmentation": arrow_shape_segmentation,
            "arrow_value_threshold": round(float(arrow_value_threshold), 3),
            "occupancy_uses_classification": False,
            "valid_letter_arrow_pair": valid_letter_arrow_pair,
            "tail_component_rejected": False,
            "median_hue": None if median_hue is None else round(median_hue, 2),
            **region_debug,
            **arrow,
        }
        if arrow["arrow_bbox"]:
            ax1, ay1, ax2, ay2 = arrow["arrow_bbox"]
            slot["arrow_bbox"] = [x1 + 3 + ax1, y1 + arrow_start + ay1, x1 + 3 + ax2, y1 + arrow_start + ay2]
        slots.append(slot)

    # A real clean prefix supplies a frame-local vertical model.  Background
    # text and animation fragments in unused tail cells sit close to the top
    # edge and have no classifier-eligible arrow; they must not lengthen the
    # physical occupied prefix.  A letter-like component near the canonical
    # baseline remains UNCERTAIN (fail closed) instead of being declared empty.
    canonical_prefix: list[dict[str, Any]] = []
    for slot in slots:
        if (
            slot["index"] != len(canonical_prefix)
            or not slot.get("valid_letter_arrow_pair")
            or slot.get("mapped_key") not in ARROW_TO_KEY.values()
        ):
            break
        canonical_prefix.append(slot)
    if canonical_prefix:
        letter_centres = [
            float(slot["selected_letter_component"]["centroid"][1])
            for slot in canonical_prefix
        ]
        letter_tops = [
            float(slot["selected_letter_component"]["bbox"][1])
            for slot in canonical_prefix
        ]
        arrow_tops = [
            float(slot["arrow_bbox"][1] - slot["bbox"][1])
            for slot in canonical_prefix
        ]
        arrow_bottoms = [
            float(slot["arrow_bbox"][3] - slot["bbox"][1])
            for slot in canonical_prefix
        ]
        slot_heights = [
            float(slot["recognition_bbox"][3] - slot["recognition_bbox"][1])
            for slot in canonical_prefix
        ]
        canonical_letter_baseline = float(np.median(letter_centres))
        canonical_letter_top = float(np.median(letter_tops))
        canonical_arrow_top = float(np.median(arrow_tops))
        canonical_arrow_bottom = float(np.median(arrow_bottoms))
        baseline_tolerance = max(3.0, float(np.median(slot_heights)) * 0.12)
        for slot in slots:
            slot.update({
                "canonical_prefix_length": len(canonical_prefix),
                "canonical_letter_baseline_y": round(
                    canonical_letter_baseline, 3
                ),
                "canonical_arrow_band": [
                    round(canonical_arrow_top, 3),
                    round(canonical_arrow_bottom, 3),
                ],
                "canonical_letter_arrow_vertical_relationship": round(
                    canonical_arrow_top - canonical_letter_baseline, 3
                ),
                "canonical_baseline_tolerance_y": round(
                    baseline_tolerance, 3
                ),
            })
        for slot in slots[len(canonical_prefix):]:
            letter = slot.get("selected_letter_component")
            if (
                not isinstance(letter, dict)
                or int(slot.get("arrow_pixel_count", 0) or 0) > 0
            ):
                continue
            centroid_y = float(letter["centroid"][1])
            letter_bottom = float(letter["bbox"][3])
            slot["tail_component_baseline_delta_y"] = round(
                centroid_y - canonical_letter_baseline, 3
            )
            is_top_edge_outlier = bool(
                centroid_y
                < canonical_letter_baseline - baseline_tolerance
                and letter_bottom
                < canonical_letter_top - max(2.0, baseline_tolerance * 0.25)
            )
            if is_top_edge_outlier:
                slot["occupancy"] = "EMPTY"
                slot["occupancy_confidence"] = 0.98
                slot["occupancy_reason"] = (
                    "unpaired_top_edge_outside_canonical_letter_baseline"
                )
                slot["tail_component_rejected"] = True

    structural_indices = [
        slot["index"] for slot in slots if slot["occupancy"] == "OCCUPIED"
    ]
    if structural_indices:
        first_structural = min(structural_indices)
        last_structural = max(structural_indices)
        for slot in slots[first_structural:last_structural + 1]:
            if (
                slot["occupancy"] != "OCCUPIED"
                and slot.get("combined_input_effect_component") is not None
            ):
                slot["occupancy"] = "OCCUPIED"
                slot["occupancy_confidence"] = 0.75
                slot["occupancy_reason"] = (
                    "continuous_structural_prefix_with_combined_input_effect"
                )

    occupied_indices = [slot["index"] for slot in slots if slot["occupancy"] == "OCCUPIED"]
    occupied_count = 0
    for slot in slots:
        if slot["occupancy"] == "OCCUPIED" and slot["index"] == occupied_count:
            occupied_count += 1
        else:
            break
    layout_conflict = any(
        slot["occupancy"] == "OCCUPIED" and slot["index"] >= occupied_count
        for slot in slots
    )
    occupied = slots[:occupied_count]
    empty_count = sum(slot["occupancy"] == "EMPTY" for slot in slots)
    uncertain_count = sum(slot["occupancy"] == "UNCERTAIN" for slot in slots)
    hues = [slot["median_hue"] for slot in occupied if slot["median_hue"] is not None]
    hue_spread = 0.0
    if len(hues) >= 2:
        hue_spread = max(
            min(abs(a - b), 180.0 - abs(a - b)) for a in hues for b in hues
        )
    panel = geometry["panel_bbox"]
    panel_hsv = hsv[panel[1]:panel[3], panel[0]:panel[2]]
    glow_ratio = float(np.mean((panel_hsv[:, :, 2] >= 215) & (panel_hsv[:, :, 1] <= 80)))
    progress_colour_detected = bool(
        occupied_count >= 2 and hue_spread >= PRESS_PROGRESS_HUE_SPREAD_MIN
    )
    input_effect_detected = bool(glow_ratio >= 0.045 or progress_colour_detected)
    effect_reasons = []
    if glow_ratio >= 0.045:
        effect_reasons.append(f"panel_glow_ratio:{glow_ratio:.4f}")
    if progress_colour_detected:
        effect_reasons.append(f"progress_colour_hue_spread:{hue_spread:.4f}")
    clean_frame_eligible = bool(
        geometry["panel_present"]
        and occupied_count > 0
        and uncertain_count == 0
        and not layout_conflict
        and not input_effect_detected
    )
    classification_complete = bool(
        occupied_count > 0 and all(slot["mapped_key"] for slot in occupied)
    )
    sequence = (
        tuple(str(slot["mapped_key"]) for slot in occupied)
        if classification_complete else ()
    )
    per_slot_confidence = tuple(float(slot["arrow_confidence"]) for slot in occupied)
    sequence_confidence = float(np.mean(per_slot_confidence)) if per_slot_confidence else 0.0
    arrow_sequence_ready = bool(
        clean_frame_eligible
        and classification_complete
        and len(sequence) == occupied_count
        and all(value >= 0.55 for value in per_slot_confidence)
        and sequence_confidence >= 0.68
    )
    return {
        "slots": slots,
        "total_slot_count": len(slots),
        "occupied_slot_count": occupied_count,
        "empty_slot_count": empty_count,
        "uncertain_slot_count": uncertain_count,
        "layout_conflict": layout_conflict,
        "input_effect_detected": input_effect_detected,
        "input_effect_reason": ";".join(effect_reasons) if effect_reasons else "none",
        "input_effect_uses_classification": False,
        "progress_colour_detected": progress_colour_detected,
        "panel_glow_ratio": round(glow_ratio, 4),
        "slot_hue_spread": round(hue_spread, 4),
        "clean_frame_eligible": clean_frame_eligible,
        "arrow_sequence": sequence,
        "arrow_sequence_confidence": round(sequence_confidence, 4),
        "arrow_sequence_ready": arrow_sequence_ready,
    }


def _save_debug_image(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    panel: tuple[int, int, int, int] | None,
    boxes: list[dict[str, Any]],
    source: Path | None,
) -> str | None:
    annotated = frame.copy()
    left, top, right, bottom = roi
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 215, 255), 2)
    cv2.putText(annotated, "press_sequence ROI", (left, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    if panel is not None:
        cv2.rectangle(annotated, (panel[0], panel[1]), (panel[2], panel[3]), (0, 220, 0), 2)
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 80, 230), 2)
        cv2.putText(annotated, item["key"], (x1, max(22, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 80, 230), 2)
    DEFAULT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stem = source.stem if source else "frame"
    destination = DEFAULT_DEBUG_DIR / f"{stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_press.png"
    return str(destination.resolve()) if cv2.imwrite(str(destination), annotated) else None


def detect_press_sequence(
    image: np.ndarray | str | Path,
    roi_config: ROIConfig | None = None,
    thresholds: ThresholdConfig | None = None,
    *,
    save_debug: bool = True,
) -> dict[str, Any]:
    """Detect PRESS panel structure and decode a frame-local sequence candidate."""
    frame, source = _load_image(image)
    config = roi_config or load_roi_config()
    active_thresholds = thresholds or load_thresholds_config()
    left, top, right, bottom = normalized_to_pixel_roi(config.rois["press_sequence"], frame.shape[1], frame.shape[0])
    crop = frame[top:bottom, left:right]
    geometry = _find_panel_geometry(crop)
    template_sets, template_errors = _load_template_sets()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    purple_ratio = float(np.mean((hsv[:, :, 0] >= 125) & (hsv[:, :, 0] <= 170) & (hsv[:, :, 1] >= 60)))
    colour_mode = "purple" if purple_ratio >= 0.003 else "mixed_live"
    templates = template_sets.get("purple" if colour_mode == "purple" else "teal", {})
    auxiliary_key_boxes = _decode_panel_slots(crop, geometry, templates)
    arrow_result = _decode_arrow_slots(crop, geometry)
    key_boxes: list[dict[str, Any]] = []
    for slot in arrow_result["slots"][:arrow_result["occupied_slot_count"]]:
        if not slot.get("mapped_key") or not slot.get("arrow_bbox"):
            continue
        x1, y1, x2, y2 = slot["arrow_bbox"]
        key_boxes.append({
            "key": slot["mapped_key"],
            "bbox": [left + x1, top + y1, left + x2, top + y2],
            "confidence": slot["arrow_confidence"],
            "top_candidates": slot["arrow_top_candidates"],
        })
    sequence = list(arrow_result["arrow_sequence"])
    sequence_text = "".join(sequence)
    glyph_confidence = (
        float(np.mean([item["confidence"] for item in auxiliary_key_boxes]))
        if auxiliary_key_boxes else 0.0
    )
    sequence_confidence = float(arrow_result["arrow_sequence_confidence"])
    panel_local = geometry["panel_bbox"]
    panel = None if panel_local is None else (
        left + panel_local[0], top + panel_local[1], left + panel_local[2], top + panel_local[3]
    )
    panel_present = bool(geometry["panel_present"])
    panel_confidence = float(geometry["panel_confidence"])
    detected = panel_present
    debug_path = _save_debug_image(frame, (left, top, right, bottom), panel, key_boxes, source) if save_debug else None
    features: list[str] = []
    if geometry["panel_candidate"]:
        features.append("press_panel_candidate")
    if panel_present:
        features.append("press_panel")
    if key_boxes:
        features.append("key_cells")
    if arrow_result["occupied_slot_count"]:
        features.append("occupied_slots")
    if arrow_result["arrow_sequence"]:
        features.append("arrow_directions")
    if templates:
        features.append("letter_templates")
    return {
        "detected": detected,
        "confidence": round(panel_confidence, 4),
        "panel_candidate": bool(geometry["panel_candidate"]),
        "panel_present": panel_present,
        "panel_qualification_reason": geometry["reason"],
        "key_box_count": len(key_boxes),
        "stable_key_box_count": 0,
        "panel_phase": (
            "PANEL_INPUT_STARTED" if arrow_result["input_effect_detected"]
            else "PANEL_CLEAN" if arrow_result["clean_frame_eligible"]
            else "PANEL_APPEARING"
        ),
        "clean_frame_eligible": arrow_result["clean_frame_eligible"],
        "input_effect_detected": arrow_result["input_effect_detected"],
        "input_effect_reason": arrow_result["input_effect_reason"],
        "selected_for_sequence": arrow_result["clean_frame_eligible"],
        "total_slot_count": arrow_result["total_slot_count"],
        "occupied_slot_count": arrow_result["occupied_slot_count"],
        "empty_slot_count": arrow_result["empty_slot_count"],
        "uncertain_slot_count": arrow_result["uncertain_slot_count"],
        "layout_conflict": arrow_result["layout_conflict"],
        "slots": arrow_result["slots"],
        "sequence": [],
        "sequence_candidate": sequence,
        "sequence_ready": False,
        "sequence_confidence": round(sequence_confidence, 4),
        "sequence_qualification_reason": (
            "earliest_clean_arrow_candidate" if arrow_result["arrow_sequence_ready"]
            else "temporal_consensus_required" if panel_present and sequence
            else "sequence_not_recoverable_from_frame" if panel_present
            else "panel_not_present"
        ),
        "sequence_text": sequence_text,
        "key_boxes": key_boxes,
        "matched_features": features,
        "debug": {
            "roi_name": "press_sequence",
            "colour_mode": colour_mode,
            "panel_bbox": list(panel) if panel else None,
            "dark_panel_ratio": round(float(geometry["dark_ratio"]), 4),
            "divider_count": int(geometry["divider_count"]),
            "grid_match_count": int(geometry["grid_match_count"]),
            "slot_count": int(geometry["slot_count"]),
            "geometry_score": round(float(geometry["geometry_score"]), 4),
            "panel_confidence": round(panel_confidence, 4),
            "glyph_confidence": round(glyph_confidence, 4),
            "arrow_sequence_confidence": arrow_result["arrow_sequence_confidence"],
            "arrow_sequence_ready": arrow_result["arrow_sequence_ready"],
            "panel_glow_ratio": arrow_result["panel_glow_ratio"],
            "slot_hue_spread": arrow_result["slot_hue_spread"],
            "template_error": template_errors.get(colour_mode),
            "template_keys": sorted(templates),
            "auxiliary_letter_sequence": "".join(
                str(item["key"]) for item in auxiliary_key_boxes
            ),
            "detector_min_confidence": active_thresholds.min_confidence_for("PRESS"),
            "debug_image_path": debug_path,
        },
    }

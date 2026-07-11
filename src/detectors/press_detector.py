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
    local_key_boxes = _decode_panel_slots(crop, geometry, templates)
    key_boxes: list[dict[str, Any]] = []
    for item in local_key_boxes:
        x1, y1, x2, y2 = item["bbox"]
        key_boxes.append({
            **item,
            "bbox": [left + x1, top + y1, left + x2, top + y2],
        })
    sequence = [item["key"] for item in key_boxes]
    sequence_text = "".join(sequence)
    sequence_confidence = float(np.mean([item["confidence"] for item in key_boxes])) if key_boxes else 0.0
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
        "sequence": [],
        "sequence_candidate": sequence,
        "sequence_ready": False,
        "sequence_confidence": round(sequence_confidence, 4),
        "sequence_qualification_reason": (
            "temporal_consensus_required" if panel_present and key_boxes
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
            "glyph_confidence": round(sequence_confidence, 4),
            "template_error": template_errors.get(colour_mode),
            "template_keys": sorted(templates),
            "detector_min_confidence": active_thresholds.min_confidence_for("PRESS"),
            "debug_image_path": debug_path,
        },
    }

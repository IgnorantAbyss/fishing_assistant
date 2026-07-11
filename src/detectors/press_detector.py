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


def _classify(glyph: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    if not templates:
        return "?", 0.0
    scores = {
        label: max(float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0]) for template in variants)
        for label, variants in templates.items()
    }
    key = max(scores, key=scores.get)
    return key, max(0.0, min(1.0, (scores[key] + 1.0) / 2.0))


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
    """Parse an eight-cell WASD panel without assuming the target sequence."""
    frame, source = _load_image(image)
    config = roi_config or load_roi_config()
    active_thresholds = thresholds or load_thresholds_config()
    left, top, right, bottom = normalized_to_pixel_roi(config.rois["press_sequence"], frame.shape[1], frame.shape[0])
    crop = frame[top:bottom, left:right]
    colour_mode, mask, local_boxes = _select_glyph_mask(crop)
    template_sets, template_errors = _load_template_sets()
    templates = template_sets.get(colour_mode, {})
    if colour_mode == "mixed_live":
        templates = template_sets.get("teal", {})
    key_boxes: list[dict[str, Any]] = []
    for box in local_boxes:
        key, confidence = _classify(
            _glyph(mask, box, tighten=colour_mode == "mixed_live"),
            templates,
        )
        key_boxes.append(
            {
                "key": key,
                "bbox": [left + box[0], top + box[1], left + box[2], top + box[3]],
                "confidence": round(confidence, 4),
            }
        )
    sequence = [item["key"] for item in key_boxes]
    sequence_text = "".join(sequence)
    panel = None
    dark_ratio = 0.0
    if local_boxes:
        panel = (
            max(0, left + min(box[0] for box in local_boxes) - 20),
            max(0, top + min(box[1] for box in local_boxes) - 16),
            min(frame.shape[1], left + max(box[2] for box in local_boxes) + 20),
            min(frame.shape[0], top + max(box[3] for box in local_boxes) + 22),
        )
        panel_gray = cv2.cvtColor(frame[panel[1]:panel[3], panel[0]:panel[2]], cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(panel_gray < 100))
    confidence = float(np.mean([item["confidence"] for item in key_boxes])) if key_boxes else 0.0
    minimum_cells = 4 if colour_mode in {"teal", "mixed_live"} else 8
    detected = (
        len(key_boxes) >= minimum_cells
        and all(key in "WASD" for key in sequence)
        and confidence >= 0.68
        and dark_ratio >= 0.14
        and bool(templates)
    )
    debug_path = _save_debug_image(frame, (left, top, right, bottom), panel, key_boxes, source) if save_debug else None
    features: list[str] = []
    if panel is not None and dark_ratio >= 0.14:
        features.append("press_panel")
    if len(key_boxes) >= minimum_cells:
        features.append("key_cells")
    if templates:
        features.append("letter_templates")
    return {
        "detected": detected,
        "confidence": round(confidence, 4),
        "sequence": sequence,
        "sequence_text": sequence_text,
        "key_boxes": key_boxes,
        "matched_features": features,
        "debug": {
            "roi_name": "press_sequence",
            "colour_mode": colour_mode,
            "dark_panel_ratio": round(dark_ratio, 4),
            "template_error": template_errors.get(colour_mode),
            "template_keys": sorted(templates),
            "detector_min_confidence": active_thresholds.min_confidence_for("PRESS"),
            "debug_image_path": debug_path,
        },
    }

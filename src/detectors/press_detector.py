"""Offline WASD sequence parser based on fixed fishing UI glyphs, not OCR."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config_loader import ROIConfig, ThresholdConfig, load_roi_config, load_thresholds_config, normalized_to_pixel_roi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_IMAGE = PROJECT_ROOT / "assets" / "reference" / "press.png"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "logs" / "press_reports"
# UI template labels for the supplied canonical static reference.  Templates
# are extracted dynamically, keeping the repository free of binary crop assets.
BOOTSTRAP_SEQUENCE = "DSDSASSD"


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


def _purple_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 0] >= 130) & (hsv[:, :, 0] <= 165) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 70)).astype(np.uint8)


def _find_letter_boxes(crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find the eight upper purple glyphs; the lower purple arrows are excluded."""
    mask = _purple_mask(crop)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for x, y, width, height, area in stats[1:count]:
        if y < int(crop.shape[0] * 0.45) or width < 8 or height < 14 or area < 70:
            continue
        candidates.append((int(x), int(y), int(x + width), int(y + height)))
    return sorted(candidates, key=lambda box: box[0])


def _glyph(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    glyph = mask[y1:y2, x1:x2]
    return cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_NEAREST)


def _load_templates(config: ROIConfig) -> tuple[dict[str, np.ndarray], str | None]:
    """Build W/A/S/D templates from the canonical press reference if available."""
    template_image = cv2.imread(str(DEFAULT_TEMPLATE_IMAGE), cv2.IMREAD_COLOR)
    if template_image is None:
        return {}, f"Template reference is unavailable: {DEFAULT_TEMPLATE_IMAGE}"
    left, top, right, bottom = normalized_to_pixel_roi(config.rois["press_sequence"], template_image.shape[1], template_image.shape[0])
    crop = template_image[top:bottom, left:right]
    mask = _purple_mask(crop)
    boxes = _find_letter_boxes(crop)
    if len(boxes) != len(BOOTSTRAP_SEQUENCE):
        return {}, f"Expected {len(BOOTSTRAP_SEQUENCE)} template cells, found {len(boxes)}"
    templates: dict[str, np.ndarray] = {}
    for label, box in zip(BOOTSTRAP_SEQUENCE, boxes, strict=True):
        templates.setdefault(label, _glyph(mask, box))
    return templates, None


def _classify(glyph: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    if not templates:
        return "?", 0.0
    scores = {
        key: float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        for key, template in templates.items()
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
) -> dict[str, Any]:
    """Parse ordered WASD glyphs from ``press_sequence`` ROI without OCR."""
    frame, source = _load_image(image)
    active_roi = roi_config or load_roi_config()
    active_thresholds = thresholds or load_thresholds_config()
    roi = normalized_to_pixel_roi(active_roi.rois["press_sequence"], frame.shape[1], frame.shape[0])
    left, top, right, bottom = roi
    crop = frame[top:bottom, left:right]
    mask = _purple_mask(crop)
    local_boxes = _find_letter_boxes(crop)
    templates, template_error = _load_templates(active_roi)
    key_boxes: list[dict[str, Any]] = []
    for box in local_boxes:
        key, confidence = _classify(_glyph(mask, box), templates)
        key_boxes.append(
            {
                "key": key,
                "bbox": [left + box[0], top + box[1], left + box[2], top + box[3]],
                "confidence": round(confidence, 4),
            }
        )
    sequence = [item["key"] for item in key_boxes]
    sequence_text = "".join(sequence)
    detected = len(key_boxes) >= 4 and all(key in "WASD" for key in sequence)
    confidence = float(np.mean([item["confidence"] for item in key_boxes])) if key_boxes else 0.0
    if len(key_boxes) == 8:
        confidence = min(1.0, confidence + 0.05)
    panel = None
    if local_boxes:
        panel = (
            left + min(box[0] for box in local_boxes) - 12,
            top + min(box[1] for box in local_boxes) - 12,
            left + max(box[2] for box in local_boxes) + 12,
            top + max(box[3] for box in local_boxes) + 18,
        )
    debug_path = _save_debug_image(frame, roi, panel, key_boxes, source)
    features: list[str] = []
    if panel is not None:
        features.append("press_panel")
    if key_boxes:
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
            "template_error": template_error,
            "template_keys": sorted(templates),
            "detector_min_confidence": active_thresholds.min_confidence_for("PRESS"),
            "debug_image_path": debug_path,
        },
    }

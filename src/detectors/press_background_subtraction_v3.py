"""Colour-invariant PRESS key-strip recognition for shadow verification.

V3 locates the ten-cell key strip first, models its dark background in LAB,
and classifies only background-subtracted binary glyph masks.  It never uses a
fixed glyph hue and never falls back to scanning the surrounding PRESS ROI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Sequence

import cv2
import numpy as np

from src.detectors.press_detector import (
    _classify_arrow,
    _classify_candidates,
    _find_panel_geometry,
    _glyph,
    _load_template_sets,
)


VALID_KEYS = frozenset("WASD")


@dataclass(frozen=True)
class PressKeyStripLocation:
    key_strip_bbox: tuple[int, int, int, int] | None
    locator_confidence: float
    slot_width: float
    slot_height: float
    slot_bboxes: tuple[tuple[int, int, int, int], ...]
    geometry_stable: bool
    rejection_reason: str | None
    progress_baseline_y: int | None = None

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["key_strip_bbox"] = (
            list(self.key_strip_bbox)
            if self.key_strip_bbox is not None else None
        )
        value["slot_bboxes"] = [list(item) for item in self.slot_bboxes]
        return value


class PressKeyStripLocator:
    """Locate the ten-cell strip without exposing surrounding UI to glyphs."""

    slot_count = 10
    _aspect_ratio = 6.45

    @classmethod
    def _vertical_grid_geometry(
        cls,
        press_roi: np.ndarray,
        *,
        fallback_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[tuple[int, int, int, int] | None, int]:
        """Fit the ten equal cells from repeated vertical grid segments."""
        if fallback_bbox is None:
            return None, 0
        height, width = press_roi.shape[:2]
        hint_x1, hint_y1, hint_x2, hint_y2 = fallback_bbox
        hint_spacing = (hint_x2 - hint_x1) / cls.slot_count
        gray = cv2.cvtColor(press_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 110)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(14, round(width * 0.025)),
            minLineLength=max(20, round(height * 0.13)),
            maxLineGap=max(3, round(height * 0.02)),
        )
        records: list[tuple[float, int, int]] = []
        horizontal: list[tuple[int, int, int]] = []
        if lines is not None:
            for raw_x1, raw_y1, raw_x2, raw_y2 in lines[:, 0]:
                x1, y1, x2, y2 = map(
                    int, (raw_x1, raw_y1, raw_x2, raw_y2)
                )
                if abs(x2 - x1) <= 2 and abs(y2 - y1) >= height * 0.12:
                    x = (x1 + x2) / 2.0
                    nearest = round((x - hint_x1) / max(1.0, hint_spacing))
                    if (
                        0 <= nearest <= cls.slot_count
                        and abs(x - (hint_x1 + nearest * hint_spacing)) <= 9
                    ):
                        records.append((x, min(y1, y2), max(y1, y2)))
                if abs(y2 - y1) <= 2 and abs(x2 - x1) >= width * 0.12:
                    horizontal.append(
                        (round((y1 + y2) / 2), min(x1, x2), max(x1, x2))
                    )
        if len(records) < 7:
            return None, 0
        clustered_x: list[float] = []
        for value in sorted(item[0] for item in records):
            if not clustered_x or value - clustered_x[-1] > 3.0:
                clustered_x.append(value)
            else:
                clustered_x[-1] = (clustered_x[-1] + value) / 2.0
        small_differences = [
            right - left
            for left, right in zip(clustered_x, clustered_x[1:])
            if hint_spacing * 0.72 <= right - left <= hint_spacing * 1.28
        ]
        spacing = (
            float(np.median(small_differences))
            if small_differences else hint_spacing
        )
        assignments: list[tuple[int, float]] = []
        for x in clustered_x:
            index = round((x - hint_x1) / max(1.0, spacing))
            if 0 <= index <= cls.slot_count:
                assignments.append((index, x - index * spacing))
        unique_indices = len({item[0] for item in assignments})
        if unique_indices < 7:
            return None, unique_indices
        left = float(np.median([item[1] for item in assignments]))
        vertical_top = int(round(float(np.median([item[1] for item in records]))))
        vertical_bottom = int(round(float(np.median([item[2] for item in records]))))
        strip_left = int(round(left))
        strip_right = int(round(left + cls.slot_count * spacing))
        relevant_horizontal = [
            row
            for row, line_left, line_right in horizontal
            if min(line_right, strip_right) - max(line_left, strip_left)
            >= (strip_right - strip_left) * 0.20
        ]
        top_rows = [
            row for row in relevant_horizontal
            if vertical_top - 6 <= row <= vertical_top + 3
        ]
        bottom_rows = [
            row for row in relevant_horizontal
            if vertical_bottom - 3 <= row <= vertical_bottom + 5
        ]
        strip_top = min(top_rows) if top_rows else vertical_top - 1
        strip_bottom = max(bottom_rows) if bottom_rows else vertical_bottom + 1
        bbox = (
            max(0, strip_left),
            max(0, strip_top),
            min(width, strip_right + 1),
            min(height, strip_bottom + 1),
        )
        return bbox, unique_indices

    @classmethod
    def _line_geometry(
        cls,
        press_roi: np.ndarray,
        *,
        fallback_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[tuple[int, int, int, int] | None, int]:
        """Find the repeated ten-cell rectangle from its long horizontal edges."""
        height, width = press_roi.shape[:2]
        gray = cv2.cvtColor(press_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 110)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(28, round(width * 0.055)),
            minLineLength=max(24, round(width * 0.45)),
            maxLineGap=max(2, round(width * 0.006)),
        )
        horizontal: list[tuple[int, int, int]] = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if abs(int(y2) - int(y1)) <= 2:
                    left, right = sorted((int(x1), int(x2)))
                    if right - left >= width * 0.50 and int(y1) >= height * 0.42:
                        horizontal.append((round((int(y1) + int(y2)) / 2), left, right))
        best: tuple[float, tuple[int, int, int, int]] | None = None
        expected_height = height * 0.30
        for top_y, top_left, top_right in horizontal:
            for bottom_y, bottom_left, bottom_right in horizontal:
                strip_height = bottom_y - top_y
                if not height * 0.23 <= strip_height <= height * 0.38:
                    continue
                overlap_left = max(top_left, bottom_left)
                overlap_right = min(top_right, bottom_right)
                strip_width = overlap_right - overlap_left
                if strip_width <= 0:
                    continue
                aspect = strip_width / max(1.0, strip_height)
                if not 5.2 <= aspect <= 7.2:
                    continue
                score = (
                    strip_width / width
                    - abs(strip_height - expected_height) / height
                    - abs(aspect - 6.45) * 0.03
                )
                bbox = (overlap_left, top_y, overlap_right + 1, bottom_y + 1)
                if best is None or score > best[0]:
                    best = (score, bbox)
        if best is None and horizontal and fallback_bbox is not None:
            line_y, line_left, line_right = max(
                horizontal, key=lambda item: item[2] - item[1]
            )
            hint_x1, hint_y1, hint_x2, hint_y2 = fallback_bbox
            strip_width = max(line_right - line_left + 1, hint_x2 - hint_x1)
            strip_height = round(strip_width / cls._aspect_ratio)
            if abs(line_y - hint_y2) <= abs(line_y - hint_y1):
                bbox = (
                    line_left,
                    line_y - strip_height + 1,
                    line_left + strip_width,
                    line_y + 1,
                )
            else:
                bbox = (
                    line_left,
                    line_y,
                    line_left + strip_width,
                    line_y + strip_height,
                )
        elif best is None:
            return None, 0
        else:
            bbox = best[1]
        x1, y1, x2, y2 = bbox
        vertical_projection = np.sum(edges[y1:y2, x1:x2] > 0, axis=0)
        expected_edges = np.linspace(0, x2 - x1 - 1, cls.slot_count + 1)
        tolerance = max(2, round((x2 - x1) / cls.slot_count * 0.10))
        matches = sum(
            bool(np.max(vertical_projection[
                max(0, round(position) - tolerance):
                min(len(vertical_projection), round(position) + tolerance + 1)
            ], initial=0) >= (y2 - y1) * 0.45)
            for position in expected_edges
        )
        return bbox, matches

    @staticmethod
    def _progress_baseline(
        press_roi: np.ndarray,
        *,
        expected_x1: int,
        expected_x2: int,
    ) -> int | None:
        hsv = cv2.cvtColor(press_roi, cv2.COLOR_BGR2HSV)
        yellow = (
            (hsv[:, :, 0] >= 10)
            & (hsv[:, :, 0] <= 45)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 65)
        )
        height, width = yellow.shape
        yellow[: int(height * 0.52)] = False
        overlap = yellow[:, max(0, expected_x1 - 4):min(width, expected_x2 + 4)]
        row_counts = np.sum(overlap, axis=1)
        minimum = max(24, round((expected_x2 - expected_x1) * 0.28))
        rows = np.flatnonzero(row_counts >= minimum)
        return int(rows[-1]) if rows.size else None

    def locate(self, press_roi: np.ndarray) -> PressKeyStripLocation:
        if (
            not isinstance(press_roi, np.ndarray)
            or press_roi.ndim != 3
            or press_roi.shape[2] != 3
            or press_roi.size == 0
        ):
            return PressKeyStripLocation(
                None, 0.0, 0.0, 0.0, (), False, "invalid_press_roi"
            )
        legacy_geometry = _find_panel_geometry(press_roi)
        legacy_bbox = legacy_geometry.get("panel_bbox")
        bbox, grid_match_count = self._vertical_grid_geometry(
            press_roi,
            fallback_bbox=(
                tuple(int(value) for value in legacy_bbox)
                if isinstance(legacy_bbox, tuple) and len(legacy_bbox) == 4
                else None
            ),
        )
        if bbox is None:
            bbox, grid_match_count = self._line_geometry(
            press_roi,
            fallback_bbox=(
                tuple(int(value) for value in legacy_bbox)
                if isinstance(legacy_bbox, tuple) and len(legacy_bbox) == 4
                else None
            ),
            )
        if bbox is None:
            return PressKeyStripLocation(
                None,
                0.0,
                0.0,
                0.0,
                (),
                False,
                "ten_slot_line_geometry_not_found",
            )
        height, width = press_roi.shape[:2]
        x1, fallback_y1, x2, fallback_y2 = (int(value) for value in bbox)
        strip_width = x2 - x1
        progress_y = self._progress_baseline(
            press_roi, expected_x1=x1, expected_x2=x2
        )
        y1, y2 = fallback_y1, fallback_y2
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + self.slot_count, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 8, min(height, y2))
        edges = np.rint(
            np.linspace(x1, x2, self.slot_count + 1)
        ).astype(int)
        slots = tuple(
            (int(edges[index]), y1, int(edges[index + 1]), y2)
            for index in range(self.slot_count)
        )
        slot_widths = [item[2] - item[0] for item in slots]
        slot_height = float(y2 - y1)
        aspect = (x2 - x1) / max(1.0, slot_height)
        stable = bool(
            len(slots) == self.slot_count
            and min(slot_widths, default=0) >= 12
            and 5.5 <= aspect <= 7.2
            and max(
                grid_match_count,
                int(legacy_geometry.get("grid_match_count", 0)),
            ) >= 8
        )
        confidence = min(
            1.0,
            0.45 * float(legacy_geometry.get("panel_confidence", 0.0))
            + 0.35 * min(
                1.0,
                max(
                    grid_match_count,
                    int(legacy_geometry.get("grid_match_count", 0)),
                ) / 10.0,
            )
            + 0.20 * (1.0 if progress_y is not None else 0.65),
        )
        return PressKeyStripLocation(
            (x1, y1, x2, y2) if stable else None,
            round(confidence, 4),
            round(float(np.mean(slot_widths)), 4),
            round(slot_height, 4),
            slots if stable else (),
            stable,
            None if stable else "ten_slot_geometry_unstable",
            progress_y,
        )


@dataclass(frozen=True)
class PressBackgroundModelResult:
    global_background_lab: tuple[float, float, float]
    per_slot_background_lab: tuple[tuple[float, float, float], ...]
    background_mad: tuple[float, float, float]
    per_slot_mad: tuple[tuple[float, float, float], ...]
    background_stability: float
    background_model_valid: bool
    rejection_reason: str | None
    local_correction_used: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class PressBackgroundModel:
    """Robust LAB model dominated by the gray-black per-cell background."""

    @staticmethod
    def _robust_centre(lab_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pixels = lab_pixels.reshape(-1, 3).astype(np.float32)
        centre = np.median(pixels, axis=0)
        distance = np.linalg.norm(pixels - centre, axis=1)
        cutoff = float(np.quantile(distance, 0.68))
        inliers = pixels[distance <= max(4.0, cutoff)]
        if len(inliers) < max(12, len(pixels) // 5):
            inliers = pixels
        centre = np.median(inliers, axis=0)
        mad = np.median(np.abs(inliers - centre), axis=0)
        return centre, mad

    def build(
        self,
        slot_inner_crops: Sequence[np.ndarray],
    ) -> PressBackgroundModelResult:
        if len(slot_inner_crops) != 10 or any(
            not isinstance(item, np.ndarray) or item.size == 0
            for item in slot_inner_crops
        ):
            return PressBackgroundModelResult(
                (0.0, 0.0, 0.0), (), (0.0, 0.0, 0.0), (),
                0.0, False, "ten_valid_slot_crops_required", False,
            )
        centres: list[np.ndarray] = []
        mads: list[np.ndarray] = []
        for crop in slot_inner_crops:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            centre, mad = self._robust_centre(lab)
            centres.append(centre)
            mads.append(mad)
        centre_array = np.vstack(centres)
        global_centre = np.median(centre_array, axis=0)
        global_mad = np.median(np.vstack(mads), axis=0)
        deviations = np.linalg.norm(centre_array - global_centre, axis=1)
        median_deviation = float(np.median(deviations))
        stability = max(0.0, min(1.0, 1.0 - median_deviation / 24.0))
        corrected: list[np.ndarray] = []
        local_used = False
        for centre in centres:
            delta = centre - global_centre
            distance = float(np.linalg.norm(delta))
            if distance <= 14.0:
                corrected.append(global_centre + np.clip(delta, -8.0, 8.0))
                local_used = local_used or distance >= 2.0
            else:
                corrected.append(global_centre.copy())
        valid = bool(
            np.all(np.isfinite(global_centre))
            and np.all(np.isfinite(global_mad))
            and stability >= 0.45
        )
        return PressBackgroundModelResult(
            tuple(round(float(value), 4) for value in global_centre),
            tuple(
                tuple(round(float(value), 4) for value in centre)
                for centre in corrected
            ),
            tuple(round(float(value), 4) for value in global_mad),
            tuple(
                tuple(round(float(value), 4) for value in mad)
                for mad in mads
            ),
            round(stability, 4),
            valid,
            None if valid else "slot_background_variation_too_large",
            local_used,
        )


@dataclass(frozen=True)
class PressForegroundResult:
    mask: np.ndarray
    distance_map: np.ndarray
    foreground_pixel_count: int
    delta_e_threshold: float
    lightness_threshold: float
    chroma_threshold: float
    local_contrast_threshold: float


class PressForegroundExtractor:
    """Extract any high-contrast glyph colour relative to the LAB background."""

    def extract(
        self,
        inner_crop: np.ndarray,
        *,
        background_lab: Sequence[float],
        background_mad: Sequence[float],
    ) -> PressForegroundResult:
        lab = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        background = np.asarray(background_lab, dtype=np.float32)
        mad = np.asarray(background_mad, dtype=np.float32)
        delta = lab - background
        delta_e = np.linalg.norm(delta, axis=2)
        lightness = np.abs(delta[:, :, 0])
        chroma = np.linalg.norm(delta[:, :, 1:3], axis=2)
        gray = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2GRAY)
        local_median = cv2.medianBlur(gray, 3)
        local_contrast = cv2.absdiff(gray, local_median).astype(np.float32)
        noise = max(1.0, float(np.median(mad)))
        chroma_noise = max(1.0, float(np.linalg.norm(mad[1:3])))
        delta_threshold = float(np.clip(10.0 + 3.2 * noise, 13.0, 34.0))
        lightness_threshold = float(
            np.clip(8.0 + 3.0 * float(mad[0]), 10.0, 30.0)
        )
        chroma_threshold = float(
            np.clip(8.0 + 2.8 * chroma_noise, 10.0, 30.0)
        )
        local_threshold = float(np.clip(6.0 + 2.0 * noise, 8.0, 18.0))
        # A strong colour/brightness departure survives even inside a solid
        # glyph.  Moderate departures additionally need a local edge, which
        # rejects smooth world-scene variation visible through the panel.
        strong_departure = (
            (delta_e >= delta_threshold * 1.55)
            | (lightness >= lightness_threshold * 1.55)
            | (chroma >= chroma_threshold * 1.55)
        )
        edge_departure = (
            (local_contrast >= local_threshold)
            & (
                (delta_e >= delta_threshold * 0.72)
                | (lightness >= lightness_threshold * 0.72)
                | (chroma >= chroma_threshold * 0.72)
            )
        )
        foreground = strong_departure | edge_departure
        mask = foreground.astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        cleaned = np.zeros_like(mask)
        minimum_area = max(2, round(mask.size * 0.0025))
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                cleaned[labels == label] = 255
        return PressForegroundResult(
            cleaned,
            np.clip(delta_e, 0, 255).astype(np.uint8),
            int(np.count_nonzero(cleaned)),
            round(delta_threshold, 4),
            round(lightness_threshold, 4),
            round(chroma_threshold, 4),
            round(local_threshold, 4),
        )


class PressBackgroundSubtractionDetectorV3:
    """Ten-slot PRESS recognizer operating only on binary foreground masks."""

    def __init__(
        self,
        *,
        locator: PressKeyStripLocator | None = None,
        background_model: PressBackgroundModel | None = None,
        foreground_extractor: PressForegroundExtractor | None = None,
    ) -> None:
        self.locator = locator or PressKeyStripLocator()
        self.background_model = background_model or PressBackgroundModel()
        self.foreground_extractor = (
            foreground_extractor or PressForegroundExtractor()
        )
        template_sets, self._template_errors = _load_template_sets()
        self._letter_templates: dict[str, list[np.ndarray]] = {
            key: [
                item
                for templates in template_sets.values()
                for item in templates.get(key, ())
            ]
            for key in VALID_KEYS
        }

    @staticmethod
    def _inner_bbox(
        slot_bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = slot_bbox
        width, height = x2 - x1, y2 - y1
        margin_x = max(2, min(4, round(width * 0.08)))
        margin_top = max(2, min(4, round(height * 0.05)))
        margin_bottom = max(5, min(8, round(height * 0.10)))
        return (
            x1 + margin_x,
            y1 + margin_top,
            x2 - margin_x,
            y2 - margin_bottom,
        )

    def _classify_slot(
        self,
        mask: np.ndarray,
        *,
        index: int,
        slot_bbox: tuple[int, int, int, int],
        inner_bbox: tuple[int, int, int, int],
        foreground: PressForegroundResult,
    ) -> dict[str, Any]:
        boolean = mask > 0
        height, width = boolean.shape[:2]
        letter_end = max(1, round(height * 0.58))
        # Native fixtures place the arrow below the letter baseline.  Starting
        # at 54% excludes the letter's anti-aliased lower edge while retaining
        # vertically shifted arrows; this is relative to the located strip.
        arrow_start = max(0, round(height * 0.54))
        arrow_end = max(arrow_start + 1, round(height * 0.96))
        letter_mask = boolean[:letter_end].astype(np.uint8)
        arrow_mask = boolean[arrow_start:arrow_end]
        arrow_distance = foreground.distance_map[arrow_start:arrow_end]
        if arrow_mask.size and np.count_nonzero(arrow_mask) >= 8:
            otsu_threshold, _ = cv2.threshold(
                arrow_distance,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            strong_arrow_mask = arrow_mask & (
                arrow_distance >= max(1.0, float(otsu_threshold))
            )
            if np.count_nonzero(strong_arrow_mask) >= 6:
                arrow_mask = strong_arrow_mask
        else:
            otsu_threshold = 0.0
        arrow = _classify_arrow(arrow_mask)
        _, _, letter_stats, _ = cv2.connectedComponentsWithStats(
            letter_mask, connectivity=8
        )
        letter_components = [
            {
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "structural": bool(
                    area >= 6 and w >= 2 and h >= 3 and w < width * 0.82
                ),
            }
            for x, y, w, h, area in letter_stats[1:]
        ]
        letter_component = max(
            (item for item in letter_components if item["structural"]),
            key=lambda item: item["area"],
            default=None,
        )
        letter_key, letter_confidence, letter_candidates = (
            _classify_candidates(
                _glyph(
                    letter_mask,
                    (0, 0, letter_mask.shape[1], letter_mask.shape[0]),
                    tighten=True,
                ),
                self._letter_templates,
            )
            if letter_mask.size and np.count_nonzero(letter_mask) >= 6
            else ("?", 0.0, [])
        )
        arrow_key = arrow.get("mapped_key")
        arrow_confidence = float(arrow.get("arrow_confidence", 0.0))
        arrow_valid = bool(
            arrow_key in VALID_KEYS and arrow_confidence >= 0.55
        )
        letter_valid = bool(
            letter_key in VALID_KEYS and float(letter_confidence) >= 0.55
        )
        letter_evidence = bool(
            isinstance(letter_component, dict)
            and int(letter_component.get("area", 0)) >= 6
        )
        letter_margin = (
            float(letter_candidates[0].get("confidence", 0.0))
            - float(letter_candidates[1].get("confidence", 0.0))
            if len(letter_candidates) >= 2 else float(letter_confidence)
        )
        conflict = bool(
            arrow_valid
            and letter_valid
            and letter_margin >= 0.12
            and str(arrow_key) != str(letter_key)
        )
        mapped_key = str(arrow_key) if arrow_valid and not conflict else None
        if arrow_valid and letter_evidence and not conflict:
            occupancy = "OCCUPIED"
            reason = (
                "binary_letter_arrow_agree"
                if letter_valid else "binary_arrow_with_letter_evidence"
            )
            confidence = min(
                1.0,
                0.72 * arrow_confidence
                + 0.28 * (float(letter_confidence) if letter_valid else 0.65),
            )
        elif (
            arrow.get("arrow_bbox") is None
            and letter_component is None
        ):
            occupancy = "EMPTY"
            reason = "no_binary_glyph_foreground"
            confidence = 1.0
        else:
            occupancy = "UNCERTAIN"
            reason = (
                "binary_letter_arrow_conflict"
                if conflict else "incomplete_binary_glyph_structure"
            )
            confidence = 0.0
        return {
            "index": index,
            "bbox": list(slot_bbox),
            "inner_bbox": list(inner_bbox),
            "occupancy": occupancy,
            "occupancy_confidence": round(
                confidence if occupancy != "UNCERTAIN" else 0.5, 4
            ),
            "possible_occupied": occupancy == "UNCERTAIN",
            "foreground_pixel_count": foreground.foreground_pixel_count,
            "components": letter_components,
            "glyph_bbox": (
                letter_component.get("bbox")
                if isinstance(letter_component, dict) else None
            ),
            "letter_candidate": (
                letter_key if letter_valid else None
            ),
            "letter_confidence": round(float(letter_confidence), 4),
            "letter_top_candidates": letter_candidates,
            "arrow_candidate": arrow.get("arrow_direction"),
            "arrow_confidence": round(arrow_confidence, 4),
            "arrow_top_candidates": arrow.get("arrow_top_candidates", []),
            "arrow_component_candidates": arrow.get(
                "arrow_component_candidates", []
            ),
            "mapped_key": mapped_key,
            "confidence": round(confidence, 4),
            "letter_arrow_conflict": conflict,
            "rejection_reasons": [] if occupancy == "OCCUPIED" else [reason],
            "occupancy_reason": reason,
            "binary_mask": mask,
            "foreground_thresholds": {
                "delta_e": foreground.delta_e_threshold,
                "lightness": foreground.lightness_threshold,
                "chroma": foreground.chroma_threshold,
                "local_contrast": foreground.local_contrast_threshold,
            },
            "letter_region": [0, 0, width, letter_end],
            "arrow_region": [0, arrow_start, width, arrow_end],
            "arrow_distance_otsu_threshold": round(
                float(otsu_threshold), 4
            ),
        }

    @staticmethod
    def _panel_effect_evidence(key_strip: np.ndarray) -> dict[str, Any]:
        """Measure broad neutral glow independently of occupancy/classification."""
        lab = cv2.cvtColor(key_strip, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0].astype(np.float32)
        chroma = np.linalg.norm(
            lab[:, :, 1:3].astype(np.float32) - 128.0, axis=2
        )
        neutral_glow = (lightness >= 215.0) & (chroma <= 18.0)
        glow_ratio = float(np.mean(neutral_glow))
        detected = glow_ratio >= 0.045
        return {
            "input_effect_detected": detected,
            "input_effect_reason": (
                f"panel_neutral_glow_ratio:{glow_ratio:.4f}"
                if detected else "none"
            ),
            "panel_neutral_glow_ratio": round(glow_ratio, 4),
            "input_effect_uses_classification": False,
        }

    def detect(self, press_roi: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        location = self.locator.locate(press_roi)
        if not location.geometry_stable or location.key_strip_bbox is None:
            return {
                "detected": False,
                "panel_candidate": False,
                "panel_present": False,
                "sequence_candidate": [],
                "slots": [],
                "locator": location.payload(),
                **location.payload(),
                "rejection_reason": location.rejection_reason,
                "processing_latency_ms": round(
                    (time.perf_counter() - started) * 1000.0, 4
                ),
            }
        raw_slots: list[np.ndarray] = []
        inner_crops: list[np.ndarray] = []
        inner_bboxes: list[tuple[int, int, int, int]] = []
        for slot_bbox in location.slot_bboxes:
            x1, y1, x2, y2 = slot_bbox
            raw_slots.append(press_roi[y1:y2, x1:x2].copy())
            inner_bbox = self._inner_bbox(slot_bbox)
            inner_bboxes.append(inner_bbox)
            ix1, iy1, ix2, iy2 = inner_bbox
            inner_crops.append(press_roi[iy1:iy2, ix1:ix2].copy())
        model = self.background_model.build(inner_crops)
        if not model.background_model_valid:
            return {
                "detected": False,
                "panel_candidate": True,
                "panel_present": True,
                "sequence_candidate": [],
                "slots": [],
                "locator": location.payload(),
                "background_model": model.payload(),
                **location.payload(),
                **model.payload(),
                "rejection_reason": model.rejection_reason,
                "processing_latency_ms": round(
                    (time.perf_counter() - started) * 1000.0, 4
                ),
            }
        slots: list[dict[str, Any]] = []
        for index, (slot_bbox, inner_bbox, inner_crop) in enumerate(zip(
            location.slot_bboxes, inner_bboxes, inner_crops, strict=True
        )):
            foreground = self.foreground_extractor.extract(
                inner_crop,
                background_lab=model.per_slot_background_lab[index],
                background_mad=model.per_slot_mad[index],
            )
            slots.append(self._classify_slot(
                foreground.mask,
                index=index,
                slot_bbox=slot_bbox,
                inner_bbox=inner_bbox,
                foreground=foreground,
            ))
        occupied_count = 0
        for slot in slots:
            if (
                slot["index"] == occupied_count
                and slot["occupancy"] == "OCCUPIED"
            ):
                occupied_count += 1
            else:
                break
        layout_conflict = any(
            slot["occupancy"] == "OCCUPIED"
            for slot in slots[occupied_count:]
        )
        occupied = slots[:occupied_count]
        classification_complete = bool(
            occupied
            and all(slot.get("mapped_key") in VALID_KEYS for slot in occupied)
        )
        sequence = (
            [str(slot["mapped_key"]) for slot in occupied]
            if classification_complete else []
        )
        trailing_empty = bool(
            all(slot["occupancy"] == "EMPTY" for slot in slots[occupied_count:])
        )
        frame_complete = bool(
            location.geometry_stable
            and classification_complete
            and not layout_conflict
            and trailing_empty
        )
        key_strip = press_roi[
            location.key_strip_bbox[1]:location.key_strip_bbox[3],
            location.key_strip_bbox[0]:location.key_strip_bbox[2],
        ].copy()
        effect = self._panel_effect_evidence(key_strip)
        clean_frame_eligible = bool(
            frame_complete and not effect["input_effect_detected"]
        )
        confidences = [float(slot["confidence"]) for slot in occupied]
        latency = (time.perf_counter() - started) * 1000.0
        return {
            "detected": True,
            "confidence": location.locator_confidence,
            "panel_candidate": True,
            "panel_present": True,
            "panel_phase": (
                "PANEL_INPUT_STARTED"
                if effect["input_effect_detected"] else
                "PANEL_CLEAN" if frame_complete else "PANEL_APPEARING"
            ),
            "clean_frame_eligible": clean_frame_eligible,
            **effect,
            "sequence_candidate": sequence,
            "sequence_ready": False,
            "sequence_confidence": round(
                float(np.mean(confidences)) if confidences else 0.0, 4
            ),
            "total_slot_count": 10,
            "occupied_slot_count": occupied_count,
            "decoded_count": len(sequence),
            "empty_slot_count": sum(
                slot["occupancy"] == "EMPTY" for slot in slots
            ),
            "uncertain_slot_count": sum(
                slot["occupancy"] == "UNCERTAIN" for slot in slots
            ),
            "layout_conflict": layout_conflict,
            "frame_complete": frame_complete,
            "slots": slots,
            "raw_slot_crops": raw_slots,
            "inner_slot_crops": inner_crops,
            "locator": location.payload(),
            "background_model": model.payload(),
            **location.payload(),
            **model.payload(),
            "key_strip_crop": key_strip,
            "processing_latency_ms": round(latency, 4),
            "rejection_reason": None if frame_complete else (
                "binary_slots_incomplete_or_ambiguous"
            ),
        }


def serializable_v3_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop pixel arrays while preserving all numeric V3 diagnostics."""
    value: dict[str, Any] = {}
    for key, item in result.items():
        if key in {"raw_slot_crops", "inner_slot_crops", "key_strip_crop"}:
            continue
        if key == "slots":
            value[key] = [
                {
                    child_key: child_value
                    for child_key, child_value in slot.items()
                    if child_key != "binary_mask"
                }
                for slot in item
            ]
        else:
            value[key] = item
    return value

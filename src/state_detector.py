"""Offline, reference-image state detection with configurable UI regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from src.config_loader import (
    DEFAULT_ROI_CONFIG_PATH,
    DEFAULT_THRESHOLDS_CONFIG_PATH,
    ROIConfig,
    ThresholdConfig,
    load_roi_config,
    load_thresholds_config,
    normalized_to_pixel_roi,
)


STATE_BY_FILENAME = {
    "idle.png": "IDLE",
    "idle2.png": "IDLE",
    "waiting.png": "WAITING",
    "ready.png": "READY",
    "ready2.png": "READY",
    "hook.png": "HOOK",
    "hook2.png": "HOOK",
    "press.png": "PRESS",
    "press2.png": "PRESS",
    "get.png": "GET",
    "get2.png": "GET",
}

# These are names of entries in config/roi.yaml, not pixel or normalized values.
ROI_NAMES_BY_STATE = {
    "IDLE": ("top_prompt", "center_space"),
    "WAITING": ("top_prompt",),
    "READY": ("top_prompt", "center_space"),
    "HOOK": ("hook_bar", "top_prompt"),
    "PRESS": ("press_sequence", "top_prompt"),
    "GET": ("get_window",),
}

FEATURE_NAMES_BY_STATE = {
    "IDLE": ("top_prompt_template", "space_button"),
    "WAITING": ("top_prompt_template",),
    "READY": ("top_prompt_template", "space_button"),
    "HOOK": ("hook_bar", "top_prompt_template"),
    "PRESS": ("press_sequence_box", "top_prompt_template"),
    "GET": ("inventory_window",),
}


@dataclass(frozen=True)
class DetectionResult:
    state: str
    confidence: float
    matched_features: list[str]
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Reference:
    path: Path
    state: str
    image: np.ndarray


class StateDetector:
    """Classify static screenshots from configurable, UI-only normalized ROIs.

    The class contains no screen capture or input-control code.  It compares an
    image with local labelled references and produces ``UNKNOWN`` instead of
    guessing when scores are below the configured safety thresholds.
    """

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        roi_config_path: str | Path | None = None,
        thresholds_config_path: str | Path | None = None,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.roi_config_path = Path(roi_config_path) if roi_config_path is not None else DEFAULT_ROI_CONFIG_PATH
        self.thresholds_config_path = (
            Path(thresholds_config_path) if thresholds_config_path is not None else DEFAULT_THRESHOLDS_CONFIG_PATH
        )
        self.roi_config: ROIConfig = load_roi_config(self.roi_config_path)
        self.thresholds: ThresholdConfig = load_thresholds_config(self.thresholds_config_path)
        self.references = self._load_references()

    def _load_references(self) -> list[_Reference]:
        references: list[_Reference] = []
        missing: list[str] = []
        for filename, state in STATE_BY_FILENAME.items():
            path = self.reference_dir / filename
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                missing.append(filename)
                continue
            references.append(_Reference(path=path, state=state, image=image))
        if not references:
            raise FileNotFoundError(f"No readable reference images were found in {self.reference_dir}")
        self.missing_references = missing
        return references

    @staticmethod
    def _crop(image: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
        height, width = image.shape[:2]
        left, top, right, bottom = normalized_to_pixel_roi(roi, width, height)
        return image[top:bottom, left:right]

    @staticmethod
    def _roi_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Return a bounded similarity for two same-purpose UI crops."""
        size = (160, 48)
        gray_a = cv2.resize(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), size)
        gray_b = cv2.resize(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY), size)
        gray_a = cv2.GaussianBlur(gray_a, (3, 3), 0)
        gray_b = cv2.GaussianBlur(gray_b, (3, 3), 0)
        template = float(cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)[0, 0])
        edges_a = cv2.Canny(gray_a, 50, 150)
        edges_b = cv2.Canny(gray_b, 50, 150)
        edge_overlap = float(
            np.count_nonzero((edges_a > 0) & (edges_b > 0))
            / max(1, np.count_nonzero((edges_a > 0) | (edges_b > 0)))
        )
        hsv_a = cv2.cvtColor(cv2.resize(image_a, size), cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(cv2.resize(image_b, size), cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [12, 8], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [12, 8], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        histogram = (float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
        return float(
            np.clip(0.65 * ((template + 1.0) / 2.0) + 0.20 * edge_overlap + 0.15 * histogram, 0.0, 1.0)
        )

    def _score_reference(self, image: np.ndarray, reference: _Reference) -> tuple[float, list[float]]:
        roi_scores = []
        for roi_name in ROI_NAMES_BY_STATE[reference.state]:
            roi = self.roi_config.rois[roi_name]
            roi_scores.append(self._roi_similarity(self._crop(image, roi), self._crop(reference.image, roi)))
        return float(np.mean(roi_scores)), roi_scores

    @staticmethod
    def _is_blank(frame: np.ndarray) -> bool:
        """Avoid treating a fully black or single-colour frame as a valid UI."""
        return frame.size == 0 or float(np.std(frame)) < 1.0

    def _unknown_result(
        self,
        raw_scores: dict[str, float],
        *,
        reason: str,
        input_path: Path | None,
        best_reference: str | None = None,
    ) -> DetectionResult:
        highest = max(raw_scores.values(), default=0.0)
        return DetectionResult(
            state="UNKNOWN",
            confidence=round(highest, 4),
            matched_features=[reason],
            debug={
                "input": str(input_path) if input_path else "<ndarray>",
                "raw_scores": {state: round(raw_scores.get(state, 0.0), 4) for state in ROI_NAMES_BY_STATE},
                "unknown_reason": reason,
                "best_reference": best_reference,
                "roi_config": str(self.roi_config.source) if self.roi_config.source else "built-in-default",
                "thresholds_config": str(self.thresholds.source) if self.thresholds.source else "built-in-default",
                "missing_references": self.missing_references,
            },
        )

    def detect(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Classify an image and include all raw state scores in debug evidence."""
        if isinstance(image, (str, Path)):
            input_path = Path(image)
            frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {input_path}")
        else:
            frame = image
            input_path = None

        if self._is_blank(frame):
            return self._unknown_result(
                {state: 0.0 for state in ROI_NAMES_BY_STATE},
                reason="blank_frame",
                input_path=input_path,
            )

        scored: list[tuple[float, _Reference, list[float]]] = [
            (*self._score_reference(frame, reference), reference) for reference in self.references
        ]
        # Reorder generated tuples into (score, reference, roi_scores) for readability.
        scored = [(score, reference, roi_scores) for score, roi_scores, reference in scored]
        scores_by_state: dict[str, list[float]] = {state: [] for state in ROI_NAMES_BY_STATE}
        for score, reference, _ in scored:
            scores_by_state[reference.state].append(score)
        raw_scores = {
            state: float(np.mean(scores)) if scores else 0.0
            for state, scores in scores_by_state.items()
        }
        best_state = max(raw_scores, key=raw_scores.get)
        best_state_score = raw_scores[best_state]
        candidates = [item for item in scored if item[1].state == best_state]
        best_reference_score, best_reference, roi_scores = max(candidates, key=lambda item: item[0])

        if best_state_score < self.thresholds.unknown_below:
            return self._unknown_result(
                raw_scores,
                reason="all_scores_below_unknown_threshold",
                input_path=input_path,
                best_reference=best_reference.path.name,
            )
        if best_state_score < self.thresholds.min_confidence_for(best_state):
            return self._unknown_result(
                raw_scores,
                reason="best_state_below_min_confidence",
                input_path=input_path,
                best_reference=best_reference.path.name,
            )

        ordered_scores = sorted(raw_scores.values(), reverse=True)
        runner_up = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
        confidence = float(
            np.clip(0.70 * best_reference_score + 0.30 * max(0.0, best_state_score - runner_up) / 0.25, 0.0, 1.0)
        )
        features = [
            FEATURE_NAMES_BY_STATE[best_state][index]
            for index, score in enumerate(roi_scores)
            if score >= self.thresholds.unknown_below
        ] or ["weak_reference_match"]
        return DetectionResult(
            state=best_state,
            confidence=round(confidence, 4),
            matched_features=features,
            debug={
                "input": str(input_path) if input_path else "<ndarray>",
                "raw_scores": {state: round(score, 4) for state, score in raw_scores.items()},
                "best_reference": best_reference.path.name,
                "best_reference_score": round(best_reference_score, 4),
                "roi_scores": {
                    FEATURE_NAMES_BY_STATE[best_state][index]: round(score, 4)
                    for index, score in enumerate(roi_scores)
                },
                "roi_config": str(self.roi_config.source) if self.roi_config.source else "built-in-default",
                "thresholds_config": str(self.thresholds.source) if self.thresholds.source else "built-in-default",
                "missing_references": self.missing_references,
            },
        )

    def detect_state(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Compatibility-friendly name for callers that prefer detect_state()."""
        return self.detect(image)


def expected_reference_images(reference_dir: str | Path) -> Iterable[tuple[Path, str]]:
    """Return labelled static-image fixtures that are currently present."""
    directory = Path(reference_dir)
    for filename, state in STATE_BY_FILENAME.items():
        path = directory / filename
        if path.exists():
            yield path, state

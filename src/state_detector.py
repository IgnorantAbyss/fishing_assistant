"""Offline state detection with static references, live UI templates, and strict fusion."""

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
from src.detectors.get_detector import detect_get_window
from src.detectors.hook_detector import detect_hook_bar
from src.detectors.press_detector import detect_press_sequence
from src.live_template_bank import DEFAULT_LIVE_TEMPLATE_ROOT, LiveTemplateBank


STATE_BY_FILENAME = {
    "idle.png": "IDLE", "idle2.png": "IDLE", "waiting.png": "WAITING",
    "ready.png": "READY", "ready2.png": "READY", "hook.png": "HOOK", "hook2.png": "HOOK",
    "press.png": "PRESS", "press2.png": "PRESS", "get.png": "GET", "get2.png": "GET",
}

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
PROMPT_STATES = ("IDLE", "WAITING", "READY")


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
    """Classify supplied images only; no capture or input-control code is present."""

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        roi_config_path: str | Path | None = None,
        thresholds_config_path: str | Path | None = None,
        live_template_root: str | Path = DEFAULT_LIVE_TEMPLATE_ROOT,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.roi_config_path = Path(roi_config_path) if roi_config_path is not None else DEFAULT_ROI_CONFIG_PATH
        self.thresholds_config_path = Path(thresholds_config_path) if thresholds_config_path is not None else DEFAULT_THRESHOLDS_CONFIG_PATH
        self.roi_config: ROIConfig = load_roi_config(self.roi_config_path)
        self.thresholds: ThresholdConfig = load_thresholds_config(self.thresholds_config_path)
        self.references = self._load_references()
        self.live_templates = LiveTemplateBank(live_template_root)

    def _load_references(self) -> list[_Reference]:
        references: list[_Reference] = []
        missing: list[str] = []
        for filename, state in STATE_BY_FILENAME.items():
            path = self.reference_dir / filename
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                missing.append(filename)
            else:
                references.append(_Reference(path, state, image))
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
        size = (160, 48)
        gray_a = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), size), (3, 3), 0)
        gray_b = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY), size), (3, 3), 0)
        template = float(cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)[0, 0])
        edges_a, edges_b = cv2.Canny(gray_a, 50, 150), cv2.Canny(gray_b, 50, 150)
        edge_overlap = float(np.count_nonzero((edges_a > 0) & (edges_b > 0)) / max(1, np.count_nonzero((edges_a > 0) | (edges_b > 0))))
        hsv_a = cv2.cvtColor(cv2.resize(image_a, size), cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(cv2.resize(image_b, size), cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [12, 8], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [12, 8], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        histogram = (float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
        return float(np.clip(0.65 * ((template + 1.0) / 2.0) + 0.20 * edge_overlap + 0.15 * histogram, 0.0, 1.0))

    @staticmethod
    def _prompt_text_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Compare bright prompt glyph structure while de-emphasising moving scenery."""
        size = (240, 80)
        gray_a = cv2.resize(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), size)
        gray_b = cv2.resize(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY), size)
        binary_a = ((gray_a > 150) * 255).astype(np.uint8)
        binary_b = ((gray_b > 150) * 255).astype(np.uint8)
        return max(0.0, float(cv2.matchTemplate(binary_a, binary_b, cv2.TM_CCOEFF_NORMED)[0, 0]))

    def _score_reference(self, image: np.ndarray, reference: _Reference) -> tuple[float, list[float]]:
        scores = [self._roi_similarity(self._crop(image, self.roi_config.rois[name]), self._crop(reference.image, self.roi_config.rois[name])) for name in ROI_NAMES_BY_STATE[reference.state]]
        return float(np.mean(scores)), scores

    def _baseline_scores(self, frame: np.ndarray) -> tuple[dict[str, float], list[tuple[float, _Reference, list[float]]]]:
        scored = []
        for reference in self.references:
            score, roi_scores = self._score_reference(frame, reference)
            scored.append((score, reference, roi_scores))
        by_state: dict[str, list[float]] = {state: [] for state in ROI_NAMES_BY_STATE}
        for score, reference, _ in scored:
            by_state[reference.state].append(score)
        return {state: float(np.mean(scores)) if scores else 0.0 for state, scores in by_state.items()}, scored

    def _live_scores(self, frame: np.ndarray) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        scores = {state: 0.0 for state in ROI_NAMES_BY_STATE}
        details: dict[str, dict[str, float]] = {}
        prompt_crop = self._crop(frame, self.roi_config.rois["top_prompt"])
        for state in PROMPT_STATES:
            templates = [template.image for template in self.live_templates.templates if template.state == state and template.roi_name == "top_prompt"]
            if not templates:
                continue
            image_score = max(self._roi_similarity(prompt_crop, template) for template in templates)
            text_score = max(self._prompt_text_similarity(prompt_crop, template) for template in templates)
            scores[state] = 0.45 * image_score + 0.55 * text_score
            details[state] = {"top_prompt_image": image_score, "top_prompt_text": text_score, "combined": scores[state]}
        for state, roi_name in (("HOOK", "hook_bar"), ("PRESS", "press_sequence"), ("GET", "get_window")):
            crop = self._crop(frame, self.roi_config.rois[roi_name])
            score = self.live_templates.score(state, roi_name, crop, self._roi_similarity)
            if score is not None:
                scores[state] = score
                details[state] = {roi_name: score}
        return scores, details

    @staticmethod
    def _is_blank(frame: np.ndarray) -> bool:
        return frame.size == 0 or float(np.std(frame)) < 1.0

    def _result(
        self,
        *,
        state: str,
        confidence: float,
        matched_features: list[str],
        raw_scores: dict[str, float],
        input_path: Path | None,
        baseline: dict[str, float],
        live: dict[str, dict[str, float]],
        components: dict[str, Any],
        reason: str | None = None,
    ) -> DetectionResult:
        debug: dict[str, Any] = {
            "input": str(input_path) if input_path else "<ndarray>",
            "raw_scores": {name: round(raw_scores[name], 4) for name in ROI_NAMES_BY_STATE},
            "static_reference_scores": {name: round(score, 4) for name, score in baseline.items()},
            "best_reference": f"{max(baseline, key=baseline.get).lower()}_reference",
            "live_template_scores": {name: {key: round(value, 4) for key, value in values.items()} for name, values in live.items()},
            "components": components,
            "roi_config": str(self.roi_config.source) if self.roi_config.source else "built-in-default",
            "thresholds_config": str(self.thresholds.source) if self.thresholds.source else "built-in-default",
            "missing_references": self.missing_references,
        }
        if reason is not None:
            debug["unknown_reason"] = reason
        return DetectionResult(state, round(float(np.clip(confidence, 0.0, 1.0)), 4), matched_features, debug)

    def detect(self, image: np.ndarray | str | Path) -> DetectionResult:
        if isinstance(image, (str, Path)):
            input_path = Path(image)
            frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {input_path}")
        else:
            frame, input_path = image, None
        if self._is_blank(frame):
            zeros = {state: 0.0 for state in ROI_NAMES_BY_STATE}
            return self._result(state="UNKNOWN", confidence=0.0, matched_features=["blank_frame"], raw_scores=zeros, input_path=input_path, baseline=zeros, live={}, components={}, reason="blank_frame")

        baseline, scored = self._baseline_scores(frame)
        live_scores, live_details = self._live_scores(frame)
        raw_scores = {state: max(baseline[state], live_scores[state]) for state in ROI_NAMES_BY_STATE}
        press = detect_press_sequence(frame, self.roi_config, self.thresholds, save_debug=False)
        hook = detect_hook_bar(frame, self.roi_config, self.thresholds, save_debug=False)
        get = detect_get_window(frame, self.roi_config, self.thresholds)
        components = {"press": press, "hook": hook, "get": get}
        if press["detected"] and press["confidence"] >= 0.68 and len(press["sequence_text"]) >= 4:
            raw_scores["PRESS"] = max(raw_scores["PRESS"], float(press["confidence"]))
            return self._result(state="PRESS", confidence=press["confidence"], matched_features=["press_panel", "key_cells", "letter_templates"], raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components)
        if hook["detected"] and hook["confidence"] >= 0.75:
            raw_scores["HOOK"] = max(raw_scores["HOOK"], float(hook["confidence"]))
            return self._result(state="HOOK", confidence=hook["confidence"], matched_features=hook["matched_features"], raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components)
        if get["detected"] and get["confidence"] >= 0.72:
            raw_scores["GET"] = max(raw_scores["GET"], float(get["confidence"]))
            return self._result(state="GET", confidence=get["confidence"], matched_features=get["matched_features"], raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components)

        baseline_best = max(baseline, key=baseline.get)
        if baseline[baseline_best] >= 0.92:
            best = next(item for item in scored if item[1].state == baseline_best and item[0] == max(value[0] for value in scored if value[1].state == baseline_best))
            features = [FEATURE_NAMES_BY_STATE[baseline_best][index] for index, score in enumerate(best[2]) if score >= self.thresholds.unknown_below] or ["static_reference_match"]
            return self._result(state=baseline_best, confidence=baseline[baseline_best], matched_features=features, raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components)

        prompt_order = sorted(PROMPT_STATES, key=lambda state: live_scores[state], reverse=True)
        prompt_state, runner = prompt_order[0], prompt_order[1]
        prompt_score, prompt_margin = live_scores[prompt_state], live_scores[prompt_state] - live_scores[runner]
        # A very strong text-template match is evidence in its own right when
        # translucent prompt backgrounds make the runner-up margin unstable.
        # This is intentionally limited to prompt states, not a global threshold.
        if prompt_score >= 0.75 or (prompt_score >= 0.62 and prompt_margin >= 0.025):
            raw_scores[prompt_state] = max(raw_scores[prompt_state], prompt_score)
            return self._result(state=prompt_state, confidence=prompt_score, matched_features=["live_top_prompt_template", "prompt_margin"], raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components)

        best_state = max(raw_scores, key=raw_scores.get)
        if baseline[best_state] >= self.thresholds.min_confidence_for(best_state):
            return self._result(state=best_state, confidence=baseline[best_state], matched_features=["static_reference_match"], raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components)
        return self._result(state="UNKNOWN", confidence=raw_scores[best_state], matched_features=["insufficient_live_or_static_evidence"], raw_scores=raw_scores, input_path=input_path, baseline=baseline, live=live_details, components=components, reason="insufficient_live_or_static_evidence")

    def detect_state(self, image: np.ndarray | str | Path) -> DetectionResult:
        return self.detect(image)


def expected_reference_images(reference_dir: str | Path) -> Iterable[tuple[Path, str]]:
    directory = Path(reference_dir)
    for filename, state in STATE_BY_FILENAME.items():
        path = directory / filename
        if path.exists():
            yield path, state

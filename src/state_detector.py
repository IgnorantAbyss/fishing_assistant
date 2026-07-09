"""Static-image state detector for the fishing mini-game.

This first-phase detector deliberately works only with screenshots and local
reference images.  It contains no screen capture or keyboard-control code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


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

# The values are normalized to make matching independent of screenshot size.
# They are local defaults for phase one; ROI calibration is intentionally left
# for the next phase.
ROI_BY_STATE = {
    "IDLE": ((0.30, 0.02, 0.70, 0.10), (0.40, 0.12, 0.56, 0.24)),
    "WAITING": ((0.30, 0.02, 0.70, 0.10),),
    "READY": ((0.30, 0.02, 0.70, 0.10), (0.40, 0.12, 0.56, 0.24)),
    "HOOK": ((0.40, 0.25, 0.60, 0.36), (0.30, 0.02, 0.70, 0.10)),
    "PRESS": ((0.40, 0.18, 0.60, 0.30), (0.30, 0.02, 0.70, 0.10)),
    "GET": ((0.76, 0.58, 0.93, 0.82),),
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
    """Classify a screenshot from UI-only reference regions.

    Each state is matched only within its fixed UI regions.  This avoids using
    the changeable world background (sea, characters, chat, and other players)
    as a state signal.  It is intentionally a reference-matching baseline,
    suitable for validating the supplied static screenshots before live screen
    capture and per-component detectors are introduced.
    """

    def __init__(self, reference_dir: str | Path) -> None:
        self.reference_dir = Path(reference_dir)
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
            raise FileNotFoundError(
                f"No readable reference images were found in {self.reference_dir}"
            )
        if missing:
            # A partial reference set remains useful during calibration.  The
            # report exposes it in debug data instead of rejecting all images.
            self.missing_references = missing
        else:
            self.missing_references = []
        return references

    @staticmethod
    def _crop(image: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = roi
        left, right = round(x1 * width), round(x2 * width)
        top, bottom = round(y1 * height), round(y2 * height)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError(f"ROI {roi} is outside the input image")
        return crop

    @staticmethod
    def _roi_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Return a robust, bounded similarity for two equal-purpose UI crops."""
        # Fixed dimensions allow different screenshot resolutions to be compared.
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

        # Template correlation is most useful for text and button shape; the
        # lighter edge and colour terms make the score less brittle in live UI.
        return float(np.clip(0.65 * ((template + 1.0) / 2.0) + 0.20 * edge_overlap + 0.15 * histogram, 0.0, 1.0))

    @staticmethod
    def _feature_name(state: str, index: int) -> str:
        names = {
            "IDLE": ("top_prompt_template", "space_button"),
            "WAITING": ("top_prompt_template",),
            "READY": ("top_prompt_template", "space_button"),
            "HOOK": ("hook_bar", "top_prompt_template"),
            "PRESS": ("press_sequence_box", "top_prompt_template"),
            "GET": ("inventory_window",),
        }
        return names[state][index]

    def _score_reference(self, image: np.ndarray, reference: _Reference) -> tuple[float, list[float]]:
        roi_scores = [
            self._roi_similarity(self._crop(image, roi), self._crop(reference.image, roi))
            for roi in ROI_BY_STATE[reference.state]
        ]
        return float(np.mean(roi_scores)), roi_scores

    def detect(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Classify a BGR image or image file and return serialisable evidence."""
        if isinstance(image, (str, Path)):
            input_path = Path(image)
            frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {input_path}")
        else:
            frame = image
            input_path = None

        scored: list[tuple[float, _Reference, list[float]]] = []
        for reference in self.references:
            score, roi_scores = self._score_reference(frame, reference)
            scored.append((score, reference, roi_scores))

        # Average by state to prevent a state with two references from receiving
        # an accidental advantage, while retaining the strongest template as evidence.
        state_scores: dict[str, list[float]] = {}
        for score, reference, _ in scored:
            state_scores.setdefault(reference.state, []).append(score)
        averaged = {state: float(np.mean(scores)) for state, scores in state_scores.items()}
        state = max(averaged, key=averaged.get)
        candidates = [item for item in scored if item[1].state == state]
        best_score, best_reference, roi_scores = max(candidates, key=lambda item: item[0])

        # Confidence is calibrated against the runner-up state.  A high raw
        # match with little separation is intentionally reported as uncertain.
        ordered = sorted(averaged.items(), key=lambda item: item[1], reverse=True)
        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = float(np.clip(0.70 * best_score + 0.30 * max(0.0, best_score - runner_up) / 0.25, 0.0, 1.0))
        features = [
            self._feature_name(state, index)
            for index, score in enumerate(roi_scores)
            if score >= 0.55
        ]
        if not features:
            features = ["weak_reference_match"]

        return DetectionResult(
            state=state,
            confidence=round(confidence, 4),
            matched_features=features,
            debug={
                "input": str(input_path) if input_path else "<ndarray>",
                "best_reference": best_reference.path.name,
                "best_reference_score": round(best_score, 4),
                "state_scores": {name: round(value, 4) for name, value in sorted(averaged.items())},
                "roi_scores": {
                    self._feature_name(state, index): round(score, 4)
                    for index, score in enumerate(roi_scores)
                },
                "missing_references": self.missing_references,
            },
        )


def expected_reference_images(reference_dir: str | Path) -> Iterable[tuple[Path, str]]:
    """Return the expected static-image fixtures that are currently present."""
    directory = Path(reference_dir)
    for filename, state in STATE_BY_FILENAME.items():
        path = directory / filename
        if path.exists():
            yield path, state

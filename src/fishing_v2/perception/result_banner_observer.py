"""Strict, hold-only observer for the reviewed Live fishing-result banner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import ResultBannerObservation


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ResultBannerConfig:
    enabled: bool
    experimental: bool
    roi: tuple[float, float, float, float]
    template: Path
    min_similarity: float
    min_dark_band_ratio: float
    min_gold_ratio: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResultBannerConfig":
        roi = data["roi"]
        template = Path(str(data["template"]))
        if not template.is_absolute():
            template = PROJECT_ROOT / template
        config = cls(
            enabled=bool(data["enabled"]),
            experimental=bool(data["experimental"]),
            roi=tuple(float(roi[name]) for name in ("x1", "y1", "x2", "y2")),
            template=template,
            min_similarity=float(data["min_similarity"]),
            min_dark_band_ratio=float(data["min_dark_band_ratio"]),
            min_gold_ratio=float(data["min_gold_ratio"]),
        )
        x1, y1, x2, y2 = config.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Result banner ROI must be normalized")
        if not (0 < config.min_similarity <= 1):
            raise ValueError("Result banner similarity must be in (0, 1]")
        return config


def _similarity(image: np.ndarray, template: np.ndarray) -> float:
    size = (320, 96)
    gray = cv2.GaussianBlur(
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size), (3, 3), 0
    )
    reference = cv2.GaussianBlur(
        cv2.resize(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), size), (3, 3), 0
    )
    correlation = (float(cv2.matchTemplate(
        gray, reference, cv2.TM_CCOEFF_NORMED
    )[0, 0]) + 1.0) / 2.0
    edges = cv2.Canny(gray, 50, 150)
    reference_edges = cv2.Canny(reference, 50, 150)
    overlap = np.count_nonzero((edges > 0) & (reference_edges > 0)) / max(
        1, np.count_nonzero((edges > 0) | (reference_edges > 0))
    )
    return float(np.clip(0.75 * correlation + 0.25 * overlap, 0.0, 1.0))


class ResultBannerObserver:
    """Experimental evidence that can delay IDLE but can never imply GET."""

    def __init__(self, config: ResultBannerConfig) -> None:
        self.config = config
        self.template = cv2.imread(str(config.template), cv2.IMREAD_COLOR)
        if self.template is None:
            raise FileNotFoundError(config.template)

    def observe(self, frame: np.ndarray, context: FrameContext) -> ResultBannerObservation:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.config.roi
        bounds = (
            round(x1 * width), round(y1 * height),
            round(x2 * width), round(y2 * height),
        )
        crop = frame[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        similarity = _similarity(crop, self.template)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_height, crop_width = gray.shape[:2]
        band = gray[
            round(crop_height * 0.25):round(crop_height * 0.95),
            round(crop_width * 0.15):round(crop_width * 0.85),
        ]
        dark_ratio = float(np.mean(band < 100))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gold_ratio = float(np.mean(
            (hsv[:, :, 0] >= 8)
            & (hsv[:, :, 0] <= 35)
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 100)
        ))
        detected = bool(
            self.config.enabled
            and similarity >= self.config.min_similarity
            and dark_ratio >= self.config.min_dark_band_ratio
            and gold_ratio >= self.config.min_gold_ratio
        )
        if not self.config.enabled:
            reason = "observer_disabled"
        elif similarity < self.config.min_similarity:
            reason = "prototype_similarity_below_strict_threshold"
        elif dark_ratio < self.config.min_dark_band_ratio:
            reason = "result_dark_band_missing"
        elif gold_ratio < self.config.min_gold_ratio:
            reason = "result_gold_divider_missing"
        else:
            reason = "reviewed_result_banner_present"
        return ResultBannerObservation(
            detected=detected,
            confidence=similarity,
            frame_index=context.frame_index,
            timestamp=context.timestamp,
            evidence={
                "roi": list(bounds),
                "similarity": round(similarity, 4),
                "dark_band_ratio": round(dark_ratio, 4),
                "gold_ratio": round(gold_ratio, 4),
                "experimental": self.config.experimental,
                "runtime_effect": "hold_result_pending_only",
                "rejection_reason": reason,
                "collect_eligible": False,
            },
        )

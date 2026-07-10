from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.detectors.get_detector import detect_get_window
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import GetObservation


class LegacyGetDetectorAdapter:
    def __init__(self, detector: Callable[..., dict[str, Any]] = detect_get_window) -> None:
        self.detector = detector

    def observe(self, frame: Any, context: FrameContext) -> GetObservation:
        try:
            result = self.detector(frame)
            return GetObservation(
                detected=bool(result.get("detected", False)),
                confidence=float(result.get("confidence", 0.0)),
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                evidence={
                    "matched_features": list(result.get("matched_features", [])),
                    "legacy_debug": dict(result.get("debug", {})),
                    "adapter": "legacy_get_detector",
                },
            )
        except Exception as exc:
            return GetObservation(
                False, 0.0, context.frame_index, context.timestamp,
                evidence={"adapter": "legacy_get_detector", "exception": f"{type(exc).__name__}: {exc}"},
            )

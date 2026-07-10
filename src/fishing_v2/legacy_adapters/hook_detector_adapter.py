from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.detectors.hook_detector import detect_hook_bar
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import HookObservation


class LegacyHookDetectorAdapter:
    def __init__(self, detector: Callable[..., dict[str, Any]] = detect_hook_bar) -> None:
        self.detector = detector

    def observe(self, frame: Any, context: FrameContext) -> HookObservation:
        try:
            result = self.detector(frame, save_debug=False)
            return HookObservation(
                detected=bool(result.get("detected", False)),
                confidence=float(result.get("confidence", 0.0)),
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                fill_ratio=result.get("fill_ratio"),
                divider_ratio=result.get("divider_ratio"),
                evidence={
                    "matched_features": list(result.get("matched_features", [])),
                    "legacy_debug": dict(result.get("debug", {})),
                    "adapter": "legacy_hook_detector",
                },
            )
        except Exception as exc:
            return HookObservation(
                False, 0.0, context.frame_index, context.timestamp,
                evidence={"adapter": "legacy_hook_detector", "exception": f"{type(exc).__name__}: {exc}"},
            )

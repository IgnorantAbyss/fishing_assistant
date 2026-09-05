from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.detectors.hook_detector import detect_hook_bar
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import HookObservation
from src.fishing_v2.runtime.hook_crossing_geometry import measure_hook_crossing_geometry


class LegacyHookDetectorAdapter:
    def __init__(self, detector: Callable[..., dict[str, Any]] = detect_hook_bar) -> None:
        self.detector = detector

    def observe(self, frame: Any, context: FrameContext) -> HookObservation:
        try:
            result = self.detector(frame, save_debug=False)
            crossing = result.get("crossing_geometry")
            if crossing is None:
                # Compatibility for injected/legacy detector implementations.
                crossing = measure_hook_crossing_geometry(frame).evidence()
            return HookObservation(
                detected=bool(result.get("detected", False)),
                confidence=float(result.get("confidence", 0.0)),
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                fill_ratio=result.get("fill_ratio"),
                divider_ratio=result.get("divider_ratio"),
                evidence={
                    "matched_features": list(result.get("matched_features", [])),
                    "bar_bbox": result.get("bar_bbox"),
                    "divider_confidence": (
                        1.0 if "divider_line" in result.get("matched_features", []) else 0.0
                    ),
                    "legacy_debug": dict(result.get("debug", {})),
                    "adapter": "legacy_hook_detector",
                    "fallback_ratio_trustworthy": False,
                    "fallback_ratio_reason": "legacy_fill_ratio_not_action_calibrated",
                    "context_score": result.get("context_score"),
                    "context_ok": result.get("context_ok"),
                    "structural_candidate": result.get("structural_candidate", False),
                    "structural_reason": result.get("structural_reason"),
                    "structural_features": list(result.get("structural_features", [])),
                    "candidate_source": result.get("candidate_source"),
                    "raw_detected": result.get(
                        "raw_detected", result.get("detected", False)
                    ),
                    **crossing,
                },
            )
        except Exception as exc:
            return HookObservation(
                False, 0.0, context.frame_index, context.timestamp,
                evidence={"adapter": "legacy_hook_detector", "exception": f"{type(exc).__name__}: {exc}"},
            )

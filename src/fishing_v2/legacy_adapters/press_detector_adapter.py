from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.detectors.press_detector import detect_press_sequence
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import PressObservation


class LegacyPressDetectorAdapter:
    def __init__(self, detector: Callable[..., dict[str, Any]] = detect_press_sequence) -> None:
        self.detector = detector

    def observe(self, frame: Any, context: FrameContext) -> PressObservation:
        try:
            result = self.detector(frame, save_debug=False)
            return PressObservation(
                detected=bool(result.get("detected", False)),
                confidence=float(result.get("confidence", 0.0)),
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                sequence=tuple(str(item) for item in result.get("sequence", [])),
                panel_candidate=bool(result.get("panel_candidate", False)),
                panel_present=bool(result.get("panel_present", False)),
                panel_qualification_reason=str(result.get("panel_qualification_reason", "")),
                key_box_count=int(result.get("key_box_count", 0)),
                stable_key_box_count=int(result.get("stable_key_box_count", 0)),
                sequence_candidate=tuple(str(item) for item in result.get("sequence_candidate", [])),
                sequence_ready=bool(result.get("sequence_ready", False)),
                sequence_confidence=float(result.get("sequence_confidence", 0.0)),
                sequence_qualification_reason=str(result.get("sequence_qualification_reason", "")),
                evidence={
                    "press_evidence_version": 2,
                    "sequence_text": str(result.get("sequence_text", "")),
                    "key_boxes": list(result.get("key_boxes", [])),
                    "panel_bbox": result.get("debug", {}).get("panel_bbox"),
                    "matched_features": list(result.get("matched_features", [])),
                    "legacy_debug": dict(result.get("debug", {})),
                    "adapter": "legacy_press_detector",
                },
            )
        except Exception as exc:
            return PressObservation(
                False, 0.0, context.frame_index, context.timestamp,
                evidence={"adapter": "legacy_press_detector", "exception": f"{type(exc).__name__}: {exc}"},
            )

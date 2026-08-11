from __future__ import annotations

from typing import Any

from src.config_loader import ROIConfig, load_roi_config
from src.detectors.press_background_subtraction_v3 import (
    PressBackgroundSubtractionDetectorV3,
    serializable_v3_result,
)
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.live.press_v3_input_effect import PressV3InputEffectTracker


def create_v3_detector() -> PressBackgroundSubtractionDetectorV3:
    return PressBackgroundSubtractionDetectorV3()


def serialize_v3_result(result: dict[str, Any]) -> dict[str, Any]:
    return serializable_v3_result(result)


def v3_result_to_observation(
    result: dict[str, Any], context: FrameContext
) -> PressObservation:
    sequence = tuple(str(item) for item in result.get("sequence_candidate", ()))
    slots = serializable_v3_result(result).get("slots", [])
    key_boxes = [
        {
            "key": item.get("mapped_key"),
            "bbox": item.get("bbox"),
            "confidence": item.get("arrow_confidence", 0.0),
            "top_candidates": item.get("arrow_top_candidates", []),
        }
        for item in slots
        if item.get("occupancy") == "OCCUPIED"
    ]
    return PressObservation(
        detected=bool(result.get("detected", False)),
        confidence=float(result.get("confidence", 0.0)),
        frame_index=context.frame_index,
        timestamp=context.timestamp,
        sequence=sequence if result.get("clean_frame_eligible") else (),
        panel_candidate=bool(result.get("panel_candidate", False)),
        panel_present=bool(result.get("panel_present", False)),
        panel_qualification_reason=str(result.get("rejection_reason") or "v3_panel"),
        key_box_count=int(result.get("decoded_count", 0)),
        stable_key_box_count=int(result.get("decoded_count", 0)),
        sequence_candidate=sequence,
        sequence_ready=False,
        sequence_confidence=float(result.get("sequence_confidence", 0.0)),
        sequence_qualification_reason=(
            "v3_complete_candidate"
            if result.get("frame_complete") else
            str(result.get("rejection_reason") or "v3_pending")
        ),
        evidence={
            "press_evidence_version": 3,
            "press_episode_id": result.get("press_episode_id"),
            "panel_disappearance_count": int(
                result.get("panel_disappearance_count", 0)
            ),
            "panel_phase": result.get("panel_phase"),
            "frame_structurally_complete": bool(
                result.get(
                    "frame_structurally_complete",
                    result.get("frame_complete", False),
                )
            ),
            "frame_clean_eligible": bool(
                result.get(
                    "frame_clean_eligible",
                    result.get("clean_frame_eligible", False),
                )
            ),
            "episode_sequence_ready": bool(
                result.get("episode_sequence_ready", False)
            ),
            "episode_input_started": bool(
                result.get("episode_input_started", False)
            ),
            "post_input_frame": bool(
                result.get("post_input_frame", False)
            ),
            "clean_frame_eligible": bool(
                result.get("clean_frame_eligible", False)
            ),
            "input_effect_detected": bool(result.get("input_effect_detected", False)),
            "total_slot_count": int(result.get("total_slot_count", 0)),
            "occupied_slot_count": int(result.get("occupied_slot_count", 0)),
            "empty_slot_count": int(result.get("empty_slot_count", 0)),
            "uncertain_slot_count": int(result.get("uncertain_slot_count", 0)),
            "layout_conflict": bool(result.get("layout_conflict", False)),
            "panel_bbox": result.get("locator", {}).get("key_strip_bbox"),
            "key_boxes": key_boxes,
            "slots": slots,
            "v3": serializable_v3_result(result),
            "adapter": "press_background_subtraction_v3",
        },
    )


class BackgroundSubtractionPressDetectorAdapter:
    """Runtime adapter; the V3 core itself accepts only the PRESS panel ROI."""

    def __init__(
        self,
        detector: PressBackgroundSubtractionDetectorV3 | None = None,
        roi_config: ROIConfig | None = None,
        input_effect_tracker: PressV3InputEffectTracker | None = None,
        panel_disappearance_frames: int = 2,
    ) -> None:
        if panel_disappearance_frames < 1:
            raise ValueError("panel_disappearance_frames must be positive")
        self.detector = detector or PressBackgroundSubtractionDetectorV3()
        self.roi_config = roi_config or load_roi_config()
        self.input_effect_tracker = (
            input_effect_tracker or PressV3InputEffectTracker()
        )
        self.panel_disappearance_frames = int(panel_disappearance_frames)
        self._episode_active = False
        self._episode_id = 0
        self._missing_panel_frames = 0

    def reset_temporal_state(self) -> None:
        """Reset only episode-relative V3 state; detector thresholds stay intact."""
        self.input_effect_tracker.reset()
        self._episode_active = False
        self._missing_panel_frames = 0

    def _apply_episode_lifecycle(
        self,
        result: dict[str, Any],
        context: FrameContext,
    ) -> dict[str, Any]:
        panel_present = bool(result.get("panel_present", False))
        if panel_present:
            if not self._episode_active:
                self.input_effect_tracker.reset()
                self._episode_active = True
                self._episode_id += 1
            self._missing_panel_frames = 0
            updated = self.input_effect_tracker.evaluate(
                result,
                frame_index=context.frame_index,
                source_capture_timestamp=context.timestamp,
            )
        else:
            updated = dict(result)
            if self._episode_active:
                self._missing_panel_frames += 1
                if (
                    self._missing_panel_frames
                    >= self.panel_disappearance_frames
                ):
                    self.input_effect_tracker.reset()
                    self._episode_active = False
            updated.update({
                "episode_input_started": (
                    self.input_effect_tracker.input_started
                    if self._episode_active else False
                ),
                "post_input_frame": False,
                "frame_clean_eligible": False,
                "clean_frame_eligible": False,
            })
        updated["press_episode_id"] = (
            self._episode_id if self._episode_active else None
        )
        updated["panel_disappearance_count"] = self._missing_panel_frames
        return updated

    def observe(self, frame: Any, context: FrameContext) -> PressObservation:
        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = self.roi_config.pixel_roi(
                "press_sequence", width, height
            )
            result = self.detector.detect(frame[y1:y2, x1:x2])
            result = self._apply_episode_lifecycle(result, context)
            return v3_result_to_observation(result, context)
        except Exception as exc:
            return PressObservation(
                False,
                0.0,
                context.frame_index,
                context.timestamp,
                evidence={
                    "press_evidence_version": 3,
                    "adapter": "press_background_subtraction_v3",
                    "exception": f"{type(exc).__name__}: {exc}",
                },
            )

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.config_loader import ROIConfig, load_roi_config
from src.detectors.press_background_subtraction_v3 import (
    PressBackgroundSubtractionDetectorV3,
    PressKeyStripLocation,
    serializable_v3_result,
)
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.live.press_v3_input_effect import PressV3InputEffectTracker


def create_v3_detector() -> PressBackgroundSubtractionDetectorV3:
    return PressBackgroundSubtractionDetectorV3()


def serialize_v3_result(result: dict[str, Any]) -> dict[str, Any]:
    return serializable_v3_result(result)


@dataclass(frozen=True)
class PressGeometryContinuityConfig:
    """Episode-local bounds expressed relative to the located slot geometry."""

    max_origin_drift_pitch_fraction: float = 0.25
    max_pitch_change_fraction: float = 0.15
    max_vertical_drift_slot_fraction: float = 0.20
    broad_panel_min_iou: float = 0.65
    locator_miss_grace_frames: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.max_origin_drift_pitch_fraction < 0.5:
            raise ValueError("origin drift bound must reject half-slot phase jumps")
        if not 0.0 <= self.max_pitch_change_fraction < 1.0:
            raise ValueError("pitch change bound must be a fraction")
        if not 0.0 <= self.max_vertical_drift_slot_fraction < 1.0:
            raise ValueError("vertical drift bound must be a fraction")
        if not 0.0 < self.broad_panel_min_iou <= 1.0:
            raise ValueError("broad panel IoU must be in (0, 1]")
        if self.locator_miss_grace_frames < 0:
            raise ValueError("locator miss grace must be non-negative")


class PressEpisodeGeometryContinuity:
    """Keep one trusted key-grid phase for one physical PRESS episode.

    The broad panel locator is used only as a continuity certificate.  It can
    preserve an already trusted grid briefly, but can never create a grid or a
    sequence by itself.
    """

    def __init__(
        self,
        config: PressGeometryContinuityConfig | None = None,
    ) -> None:
        self.config = config or PressGeometryContinuityConfig()
        self._anchor: PressKeyStripLocation | None = None
        self._locator_miss_streak = 0

    @property
    def anchor(self) -> PressKeyStripLocation | None:
        return self._anchor

    @property
    def locator_miss_streak(self) -> int:
        return self._locator_miss_streak

    def reset(self) -> None:
        self._anchor = None
        self._locator_miss_streak = 0

    @staticmethod
    def _bbox_iou(
        left: tuple[int, int, int, int] | None,
        right: tuple[int, int, int, int] | None,
    ) -> float:
        if left is None or right is None:
            return 0.0
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = left_area + right_area - intersection
        return float(intersection / union) if union > 0 else 0.0

    def resolve(
        self,
        candidate: PressKeyStripLocation,
    ) -> tuple[PressKeyStripLocation, dict[str, Any]]:
        anchor = self._anchor
        candidate_valid = bool(
            candidate.geometry_stable
            and candidate.key_strip_bbox is not None
            and len(candidate.slot_bboxes) == 10
        )
        broad_iou = self._bbox_iou(
            anchor.broad_panel_bbox if anchor is not None else None,
            candidate.broad_panel_bbox,
        )
        same_broad_panel = bool(
            anchor is not None
            and candidate.broad_panel_present
            and broad_iou >= self.config.broad_panel_min_iou
        )
        diagnostic: dict[str, Any] = {
            "anchor_active": anchor is not None,
            "current_locator_valid": candidate_valid,
            "broad_panel_candidate": candidate.broad_panel_candidate,
            "broad_panel_present": candidate.broad_panel_present,
            "broad_panel_iou": round(broad_iou, 4),
            "same_broad_panel": same_broad_panel,
            "candidate_bbox": (
                list(candidate.key_strip_bbox)
                if candidate.key_strip_bbox is not None else None
            ),
            "anchor_bbox": (
                list(anchor.key_strip_bbox)
                if anchor is not None and anchor.key_strip_bbox is not None
                else None
            ),
            "candidate_pitch": candidate.slot_width,
            "anchor_pitch": anchor.slot_width if anchor is not None else None,
            "origin_delta_pitch_fraction": None,
            "pitch_change_fraction": None,
            "vertical_delta_slot_fraction": None,
            "locator_miss_streak": self._locator_miss_streak,
            "locator_miss_grace_frames": self.config.locator_miss_grace_frames,
            "using_anchor": False,
            "decision": "current_geometry_no_anchor",
            "rejection_reason": None,
        }
        if anchor is None:
            self._locator_miss_streak = 0 if candidate_valid else 1
            diagnostic["locator_miss_streak"] = self._locator_miss_streak
            return candidate, diagnostic

        if candidate_valid:
            self._locator_miss_streak = 0
            assert anchor.key_strip_bbox is not None
            assert candidate.key_strip_bbox is not None
            origin_delta = (
                candidate.key_strip_bbox[0] - anchor.key_strip_bbox[0]
            ) / max(1.0, anchor.slot_width)
            pitch_delta = abs(candidate.slot_width - anchor.slot_width) / max(
                1.0, anchor.slot_width
            )
            vertical_delta = (
                candidate.key_strip_bbox[1] - anchor.key_strip_bbox[1]
            ) / max(1.0, anchor.slot_height)
            diagnostic.update({
                "origin_delta_pitch_fraction": round(origin_delta, 4),
                "pitch_change_fraction": round(pitch_delta, 4),
                "vertical_delta_slot_fraction": round(vertical_delta, 4),
                "locator_miss_streak": 0,
            })
            within_episode_drift = bool(
                abs(origin_delta)
                <= self.config.max_origin_drift_pitch_fraction
                and pitch_delta <= self.config.max_pitch_change_fraction
                and abs(vertical_delta)
                <= self.config.max_vertical_drift_slot_fraction
            )
            if same_broad_panel and not within_episode_drift:
                diagnostic.update({
                    "using_anchor": True,
                    "decision": "anchor_preserved_after_grid_phase_jump",
                    "rejection_reason": "current_grid_phase_jump",
                })
                return anchor, diagnostic
            if same_broad_panel:
                diagnostic["decision"] = (
                    "current_geometry_within_episode_drift"
                )
                return candidate, diagnostic
            diagnostic.update({
                "decision": "current_geometry_without_episode_continuity",
                "rejection_reason": "broad_panel_continuity_not_confirmed",
            })
            return replace(
                candidate,
                key_strip_bbox=None,
                slot_bboxes=(),
                geometry_stable=False,
                rejection_reason="broad_panel_continuity_not_confirmed",
            ), diagnostic

        self._locator_miss_streak += 1
        diagnostic["locator_miss_streak"] = self._locator_miss_streak
        if (
            same_broad_panel
            and self._locator_miss_streak
            <= self.config.locator_miss_grace_frames
        ):
            diagnostic.update({
                "using_anchor": True,
                "decision": "anchor_preserved_during_locator_miss",
                "rejection_reason": candidate.rejection_reason,
            })
            return anchor, diagnostic
        diagnostic.update({
            "decision": (
                "locator_miss_grace_exhausted"
                if same_broad_panel else "broad_panel_not_continuous"
            ),
            "rejection_reason": candidate.rejection_reason,
        })
        return candidate, diagnostic

    def consider_anchor(
        self,
        candidate: PressKeyStripLocation,
        result: dict[str, Any],
    ) -> bool:
        if self._anchor is not None:
            return False
        if not (
            candidate.geometry_stable
            and candidate.key_strip_bbox is not None
            and len(candidate.slot_bboxes) == 10
            and bool(result.get("frame_structurally_complete", False))
        ):
            return False
        self._anchor = candidate
        self._locator_miss_streak = 0
        return True


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
        geometry_continuity: PressEpisodeGeometryContinuity | None = None,
        panel_disappearance_frames: int = 2,
    ) -> None:
        if panel_disappearance_frames < 1:
            raise ValueError("panel_disappearance_frames must be positive")
        self.detector = detector or PressBackgroundSubtractionDetectorV3()
        self.roi_config = roi_config or load_roi_config()
        self.input_effect_tracker = (
            input_effect_tracker or PressV3InputEffectTracker()
        )
        self.geometry_continuity = (
            geometry_continuity or PressEpisodeGeometryContinuity()
        )
        self.panel_disappearance_frames = int(panel_disappearance_frames)
        self._episode_active = False
        self._episode_id = 0
        self._missing_panel_frames = 0

    def reset_temporal_state(self) -> None:
        """Reset only episode-relative V3 state; detector thresholds stay intact."""
        self.input_effect_tracker.reset()
        self.geometry_continuity.reset()
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
                    self.geometry_continuity.reset()
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
        continuity = updated.get("geometry_continuity")
        if isinstance(continuity, dict):
            continuity["adapter_panel_missing_frames"] = (
                self._missing_panel_frames
            )
            continuity["episode_disappearance_confirmed"] = bool(
                not self._episode_active
                and self._missing_panel_frames
                >= self.panel_disappearance_frames
            )
        return updated

    def observe(self, frame: Any, context: FrameContext) -> PressObservation:
        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = self.roi_config.pixel_roi(
                "press_sequence", width, height
            )
            press_roi = frame[y1:y2, x1:x2]
            candidate = self.detector.locator.locate(press_roi)
            location, continuity = self.geometry_continuity.resolve(candidate)
            result = self.detector.detect(press_roi, location=location)
            anchor_created = self.geometry_continuity.consider_anchor(
                candidate,
                result,
            )
            continuity["anchor_created"] = anchor_created
            if anchor_created:
                continuity["anchor_active"] = True
                continuity["anchor_bbox"] = list(
                    candidate.key_strip_bbox or ()
                )
                continuity["anchor_pitch"] = candidate.slot_width
                continuity["decision"] = "episode_anchor_created"
            result["geometry_continuity"] = continuity
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

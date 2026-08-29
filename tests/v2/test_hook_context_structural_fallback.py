from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import src.detectors.hook_detector as hook_detector
from src.config_loader import ROIConfig
from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.observations import HookObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.live.hook_critical_loop import LatestHookFrameSlot, HookROIFrame
from src.fishing_v2.live.hook_pending_timeout_evidence import (
    HookPendingTimeoutEvidenceRecorder,
)


PRECISE_ROI_CONFIG = ROIConfig(
    screen_reference=(614, 58),
    rois={
        "hook_bar_precise": (0.0, 0.0, 1.0, 1.0),
        "hook_prompt": (0.0, 0.0, 1.0, 1.0),
    },
    source=None,
)


def _structural_frame(
    *,
    fill: bool = True,
    cyan: bool = True,
    divider: bool = True,
) -> np.ndarray:
    frame = np.zeros((58, 614, 3), dtype=np.uint8)
    if fill:
        frame[16:41, 109:357] = (0, 0, 255)
    if divider:
        frame[12:45, 357:360] = (255, 255, 255)
    if cyan:
        cyan_start = 109 if not fill else 360 if divider else 357
        frame[16:41, cyan_start:500] = (120, 120, 0)
        if divider:
            frame[12:45, 357:360] = (255, 255, 255)
    return frame


def _detect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    context_score: float,
    frame: np.ndarray | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(
        hook_detector, "_live_context_score",
        lambda crop, *, precise: context_score,
    )
    monkeypatch.setattr(hook_detector, "_hook_prompt_score", lambda crop: None)
    return hook_detector.detect_hook_bar(
        _structural_frame() if frame is None else frame,
        roi_config=PRECISE_ROI_CONFIG,
        save_debug=False,
    )


def _qualified_bundle(
    result: dict[str, object],
    *,
    frame_index: int,
    timestamp: float,
) -> tuple[ObservationBundle, object]:
    raw = HookObservation(
        detected=bool(result["detected"]),
        confidence=float(result["confidence"]),
        frame_index=frame_index,
        timestamp=timestamp,
        fill_ratio=float(result["fill_ratio"]),
        divider_ratio=float(result["divider_ratio"]),
        evidence={
            "matched_features": list(result["matched_features"]),
            "candidate_source": result["candidate_source"],
            "context_score": result["context_score"],
            "context_ok": result["context_ok"],
            "structural_candidate": result["structural_candidate"],
        },
    )
    qualified, qualification = DetectorEvidenceQualifier().qualify_hook(
        raw, DetectorActivationMode.BURST
    )
    return ObservationBundle(frame_index, timestamp, None, qualified), qualification


def _transition_timing(result: dict[str, object]) -> tuple[str, str, int]:
    fsm = FishingFSM(
        FSMConfig(stable_frames=2),
        initial_state=RuntimeState.HOOK_PENDING,
        initial_timestamp=0.0,
    )
    reasons: list[str] = []
    transition_frame = 0
    for frame_index, timestamp in ((1, 0.025), (2, 0.050)):
        bundle, qualification = _qualified_bundle(
            result, frame_index=frame_index, timestamp=timestamp
        )
        assert qualification.qualified_detected is True
        evidence = ObservationFusion().fuse(bundle, fsm.state)
        outcome = fsm.advance(
            evidence, timestamp, bundle, recorded_observation=True
        )
        reasons.append(outcome.transition_reason)
        if outcome.next_state == RuntimeState.HOOK:
            transition_frame = frame_index
        assert outcome.action_request.intent == ActionIntent.NONE
    return reasons[0], reasons[1], transition_frame


def test_low_context_live_anomaly_shape_is_raw_structural_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _detect(monkeypatch, context_score=0.31)

    assert result["detected"] is True
    assert result["raw_detected"] is True
    assert result["context_ok"] is False
    assert result["debug"]["raw_values"]["context_floor"] == 0.36
    assert result["structural_candidate"] is True
    assert result["structural_reason"] == "strong_hook_structural_evidence"
    assert result["candidate_source"] == "structural"
    assert {"bar_fill", "divider_line"} <= set(result["matched_features"])
    assert result["should_press_space"] is False


def test_normal_context_path_keeps_same_frame_candidate_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_result = _detect(monkeypatch, context_score=0.80)
    structural_result = _detect(monkeypatch, context_score=0.31)

    assert context_result["detected"] is True
    assert context_result["candidate_source"] == "both"
    assert _transition_timing(context_result) == (
        "candidate_not_stable", "stable_strong_hook_evidence", 2
    )
    assert _transition_timing(structural_result) == (
        "candidate_not_stable", "stable_strong_hook_evidence", 2
    )


def test_normal_context_path_does_not_require_structural_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _detect(
        monkeypatch,
        context_score=0.80,
        frame=_structural_frame(divider=False),
    )

    assert result["detected"] is True
    assert result["context_ok"] is True
    assert result["structural_candidate"] is False
    assert result["candidate_source"] == "context"


@pytest.mark.parametrize(
    ("frame_factory", "reason"),
    [
        (lambda: _structural_frame(cyan=False), "divider_line_missing"),
        (lambda: _structural_frame(divider=False), "divider_line_missing"),
        (lambda: _structural_frame(fill=False), "bar_fill_missing"),
    ],
)
def test_low_context_weak_or_partial_structure_abstains(
    monkeypatch: pytest.MonkeyPatch,
    frame_factory: Callable[[], np.ndarray],
    reason: str,
) -> None:
    result = _detect(
        monkeypatch, context_score=0.31, frame=frame_factory()
    )

    assert result["detected"] is False
    assert result["structural_candidate"] is False
    assert result["structural_reason"] == reason
    assert result["candidate_source"] is None


def test_low_context_conflicting_geometry_abstains() -> None:
    mask = np.ones((58, 614), dtype=np.uint8)
    candidate, reason, _ = hook_detector._strong_structural_candidate(
        precise_bar=True,
        shape_geometry_ok=True,
        local_bar=(100, 16, 500, 41),
        local_fill=(90, 16, 357, 41),
        local_divider=357,
        red_mask=mask,
        cyan_mask=mask,
    )

    assert candidate is False
    assert reason == "divider_fill_geometry_conflict"


def test_structural_path_adds_no_secondary_temporal_counter() -> None:
    assert not hasattr(hook_detector, "structural_stability_count")
    assert not hasattr(hook_detector, "structural_confirmed_after_n_frames")


def test_latest_hook_slot_still_drops_stale_frame_before_qualification() -> None:
    slot = LatestHookFrameSlot()
    slot.publish(HookROIFrame(91, 1.573, _structural_frame()))
    slot.publish(HookROIFrame(92, 1.602, _structural_frame()))

    latest = slot.take_latest()

    assert latest is not None
    assert latest.frame_index == 92
    assert slot.stale_frames_dropped == 1


def test_structural_detection_does_not_bypass_action_crossing_or_one_shot() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=2),
        initial_state=RuntimeState.HOOK_PENDING,
        initial_timestamp=0.0,
    )

    def bundle(frame_index: int, endpoint: float) -> ObservationBundle:
        timestamp = frame_index * 0.025
        hook = HookObservation(
            True,
            1.0,
            frame_index,
            timestamp,
            fill_ratio=0.88,
            evidence={
                "matched_features": [
                    "hook_bar_rect", "bar_fill", "divider_line"
                ],
                "candidate_source": "structural",
                "crossing_geometry_version": 1,
                "divider_line_detected": True,
                "divider_line_x": 1330.0,
                "divider_confidence": 1.0,
                "fill_endpoint_x": endpoint,
            },
        )
        return ObservationBundle(frame_index, timestamp, None, hook)

    for frame_index in (1, 2):
        current = bundle(frame_index, 1339.0)
        evidence = ObservationFusion().fuse(current, fsm.state)
        result = fsm.advance(
            evidence,
            current.timestamp,
            current,
            recorded_observation=True,
            hook_action_observation=current.hook,
        )
        assert result.action_request.intent == ActionIntent.NONE

    crossed = bundle(3, 1341.0)
    first = fsm.advance(
        ObservationFusion().fuse(crossed, fsm.state),
        crossed.timestamp,
        crossed,
        recorded_observation=True,
        hook_action_observation=crossed.hook,
    )
    repeated = bundle(4, 1400.0)
    second = fsm.advance(
        ObservationFusion().fuse(repeated, fsm.state),
        repeated.timestamp,
        repeated,
        recorded_observation=True,
        hook_action_observation=repeated.hook,
    )

    assert fsm.state == RuntimeState.HOOK
    assert first.action_request.intent == ActionIntent.HOOK_ACTION
    assert second.action_request.intent == ActionIntent.NONE


def test_detector_has_no_capture_or_sleep_dependency() -> None:
    names = set(hook_detector.detect_hook_bar.__code__.co_names)
    assert "capture" not in names
    assert "sleep" not in names


def test_hook_timeout_manifest_payload_exposes_candidate_path() -> None:
    observation = HookObservation(
        True,
        1.0,
        91,
        1.573,
        fill_ratio=0.88,
        evidence={
            "matched_features": [
                "hook_bar_rect", "bar_fill", "divider_line"
            ],
            "raw_detected": True,
            "context_score": 0.3141,
            "context_ok": False,
            "structural_candidate": True,
            "structural_reason": "strong_hook_structural_evidence",
            "structural_features": [
                "bar_shape_geometry", "divider_fill_geometry"
            ],
            "candidate_source": "structural",
        },
    )

    payload = HookPendingTimeoutEvidenceRecorder._raw_payload(
        observation, evaluated=True
    )

    assert payload["raw_detected"] is True
    assert payload["context_score"] == 0.3141
    assert payload["context_ok"] is False
    assert payload["structural_candidate"] is True
    assert payload["candidate_source"] == "structural"


def test_hook_timeout_and_roi_contracts_are_unchanged() -> None:
    assert FSMConfig().hook_pending_timeout_sec == 3.0
    assert tuple(PRECISE_ROI_CONFIG.rois) == (
        "hook_bar_precise", "hook_prompt"
    )

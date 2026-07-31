from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.observations import HookObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM


def _evidence(frame: int = 1) -> StateEvidence:
    return StateEvidence(
        {RuntimeState.HOOK: 0.98}, ("hook",), (), RuntimeState.HOOK,
        0.98, "strong_hook_evidence", frame, frame * 0.2,
    )


def _bundle(
    frame: int,
    *,
    detected: bool = True,
    fill_ratio: float | None,
    divider_ratio: float | None = None,
    features: tuple[str, ...] = ("hook_bar_rect", "bar_fill"),
    crossing: dict | None = None,
) -> ObservationBundle:
    timestamp = frame * 0.2
    hook = HookObservation(
        detected, 0.98 if detected else 0.0, frame, timestamp,
        fill_ratio=fill_ratio,
        divider_ratio=divider_ratio,
        evidence={
            "matched_features": list(features),
            "bar_bbox": [100, 200, 500, 230],
            **(crossing or {}),
        },
    )
    return ObservationBundle(frame, timestamp, None, hook, None, None)


def test_fill_before_detected_divider_does_not_propose_action() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)

    result = fsm.advance(
        _evidence(), 0.2,
        _bundle(1, fill_ratio=0.70, divider_ratio=0.80, features=(
            "hook_bar_rect", "bar_fill", "divider_line",
        )),
        recorded_observation=True,
    )

    assert result.action_request.intent == ActionIntent.NONE


def test_fill_above_point_eighty_five_uses_fallback_without_upper_bound() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)

    result = fsm.advance(
        _evidence(), 0.2,
        _bundle(1, fill_ratio=0.92),
        recorded_observation=True,
    )

    assert result.action_request.intent == ActionIntent.HOOK_ACTION


def test_hook_action_is_proposed_once_even_when_never_applied() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    first = fsm.advance(
        _evidence(), 0.2, _bundle(1, fill_ratio=0.72),
        recorded_observation=True,
    )
    second = fsm.advance(
        _evidence(2), 0.4, _bundle(2, fill_ratio=0.75),
        recorded_observation=True,
    )

    assert first.action_request.intent == ActionIntent.HOOK_ACTION
    assert second.action_request.intent == ActionIntent.NONE


def test_hook_proposal_can_be_released_before_emission_then_consumed_once() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.HOOK,
    )
    crossing = {
        "crossing_geometry_version": 1,
        "divider_line_detected": True,
        "divider_line_x": 300.0,
        "divider_confidence": 1.0,
        "fill_endpoint_x": 320.0,
    }
    first = fsm.advance(
        _evidence(),
        0.2,
        _bundle(1, fill_ratio=0.50, crossing=crossing),
        recorded_observation=True,
    )

    assert first.action_request.intent == ActionIntent.HOOK_ACTION
    assert fsm.hook_opportunity_lifecycle == "reserved"
    assert fsm.hook_episode_active is True
    assert fsm.release_hook_proposal_for_retry(first.action_request) is True
    assert fsm.hook_opportunity_lifecycle == "available"

    second = fsm.advance(
        _evidence(2),
        0.4,
        _bundle(2, fill_ratio=0.55, crossing=crossing),
        recorded_observation=True,
    )
    assert second.action_request.intent == ActionIntent.HOOK_ACTION
    assert fsm.mark_hook_emission_started(second.action_request) is True
    assert fsm.hook_opportunity_lifecycle == "consumed"
    assert fsm.hook_episode_active is False

    committed = fsm.commit_action(second.action_request, 0.4)
    assert committed.action_applied is True
    assert committed.next_state == RuntimeState.RESULT_PENDING


def test_detected_bar_is_not_ready_without_positive_fill() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(), 0.2,
        _bundle(1, fill_ratio=0.0, features=("hook_bar_rect",)),
        recorded_observation=True,
    )
    assert result.action_request.intent == ActionIntent.NONE


def test_explicit_divider_blocks_ratio_fallback_until_cyan_fill_exists() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    crossing = {
        "crossing_geometry_version": 1,
        "divider_line_detected": True,
        "divider_line_x": 300.0,
        "divider_confidence": 1.0,
        "fill_endpoint_x": None,
    }
    result = fsm.advance(
        _evidence(), 0.2,
        _bundle(1, fill_ratio=1.0, divider_ratio=0.5, crossing=crossing),
        recorded_observation=True,
    )
    assert result.action_request.intent == ActionIntent.NONE


def test_explicit_divider_margin_proposes_action_once() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    crossing = {
        "crossing_geometry_version": 1,
        "divider_line_detected": True,
        "divider_line_x": 300.0,
        "divider_confidence": 1.0,
        "fill_endpoint_x": 310.0,
    }
    first = fsm.advance(
        _evidence(), 0.2,
        _bundle(1, fill_ratio=0.50, divider_ratio=0.75, crossing=crossing),
        recorded_observation=True,
    )
    second = fsm.advance(
        _evidence(2), 0.4,
        _bundle(2, fill_ratio=0.95, divider_ratio=0.75, crossing={
            **crossing, "fill_endpoint_x": 380.0,
        }),
        recorded_observation=True,
    )
    assert first.action_request.intent == ActionIntent.HOOK_ACTION
    assert first.action_request.payload["divider_margin_passed"] is True
    assert second.action_request.intent == ActionIntent.NONE


def test_fallback_threshold_is_inclusive_and_has_no_upper_bound() -> None:
    for ratio in (0.70, 0.99):
        fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
        result = fsm.advance(
            _evidence(), 0.2, _bundle(1, fill_ratio=ratio),
            recorded_observation=True,
        )
        assert result.action_request.intent == ActionIntent.HOOK_ACTION
        assert result.action_request.payload["fallback_used"] is True


def test_failed_action_geometry_uses_ratio_fallback_not_legacy_divider() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(), 0.2,
        _bundle(
            1,
            fill_ratio=0.70,
            divider_ratio=0.90,
            features=("hook_bar_rect", "bar_fill", "divider_line"),
            crossing={
                "crossing_geometry_version": 1,
                "divider_line_detected": False,
                "divider_line_x": None,
                "divider_confidence": 0.0,
                "fill_endpoint_x": None,
                "fallback_ratio_trustworthy": True,
            },
        ),
        recorded_observation=True,
    )
    assert result.action_request.intent == ActionIntent.HOOK_ACTION
    assert result.action_request.payload["fallback_used"] is True


def test_bar_disappearance_never_proposes_hook_action() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(), 0.2,
        _bundle(1, detected=False, fill_ratio=None, features=()),
        recorded_observation=True,
    )
    assert result.action_request.intent == ActionIntent.NONE


def test_sampling_gap_is_classified_as_sampling_not_calibration() -> None:
    from tools.hook_threshold_crossing_diagnostics import classify_missing_observation_band

    assert classify_missing_observation_band([0.61, 0.89]) == "sampling_skipped_observation_band"

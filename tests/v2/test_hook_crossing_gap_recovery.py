from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.observations import HookObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM


def _evidence(frame: int) -> StateEvidence:
    return StateEvidence(
        {RuntimeState.HOOK: 0.98}, ("hook",), (), RuntimeState.HOOK,
        0.98, "strong_hook_evidence", frame, frame * 0.2,
    )


def _bundle(
    frame: int,
    *,
    qualified: bool,
    endpoint: float | None,
    divider_detected: bool = True,
    fill_ratio: float | None = 1.0,
    fallback_ratio_trustworthy: bool = False,
    features: tuple[str, ...] = ("hook_bar_rect", "bar_fill", "divider_line"),
) -> ObservationBundle:
    timestamp = frame * 0.2
    observation = HookObservation(
        qualified,
        1.0 if qualified else 0.54,
        frame,
        timestamp,
        fill_ratio=fill_ratio,
        divider_ratio=0.51 if divider_detected else None,
        evidence={
            "matched_features": list(features),
            "crossing_geometry_version": 1,
            "divider_line_detected": divider_detected,
            "divider_line_x": 1330.0 if divider_detected else None,
            "divider_confidence": 1.0 if divider_detected else 0.0,
            "fill_endpoint_x": endpoint,
            "fallback_ratio_trustworthy": fallback_ratio_trustworthy,
        },
    )
    return ObservationBundle(frame, timestamp, None, observation, None, None)


def test_current_safe_geometry_recovers_after_qualified_crossing_gap() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.HOOK,
        initial_timestamp=73.4,
    )
    before = fsm.advance(
        _evidence(368), 73.6,
        _bundle(368, qualified=True, endpoint=1335.0),
        recorded_observation=True,
    )
    recovered = fsm.advance(
        _evidence(369), 73.8,
        _bundle(369, qualified=False, endpoint=1373.0),
        recorded_observation=True,
    )

    assert before.action_request.intent == ActionIntent.NONE
    assert recovered.action_request.intent == ActionIntent.HOOK_ACTION
    assert recovered.action_request.payload["hook_episode_active"] is True
    assert recovered.action_request.payload["current_hook_geometry_is_usable"] is True
    assert recovered.action_request.payload["qualified_active_hook_bar"] is False


def test_recovery_is_level_triggered_and_does_not_require_crossing_edge() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    recovered = fsm.advance(
        _evidence(10), 2.0,
        _bundle(10, qualified=False, endpoint=1380.0),
        recorded_observation=True,
    )

    assert recovered.action_request.intent == ActionIntent.HOOK_ACTION


def test_untrusted_legacy_ratio_one_cannot_drive_fallback() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(1), 0.2,
        _bundle(
            1,
            qualified=True,
            endpoint=None,
            divider_detected=False,
            fill_ratio=1.0,
            fallback_ratio_trustworthy=False,
        ),
        recorded_observation=True,
    )

    assert result.action_request.intent == ActionIntent.NONE
    assert result.transition_reason == "fallback_ratio_not_trustworthy"


def test_trusted_ratio_fallback_remains_level_triggered_without_upper_bound() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(1), 0.2,
        _bundle(
            1,
            qualified=True,
            endpoint=None,
            divider_detected=False,
            fill_ratio=0.92,
            fallback_ratio_trustworthy=True,
        ),
        recorded_observation=True,
    )

    assert result.action_request.intent == ActionIntent.HOOK_ACTION
    assert result.action_request.payload["fallback_used"] is True


def test_rectangle_only_frame_cannot_start_or_recover_hook_action() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(1), 0.2,
        _bundle(
            1,
            qualified=False,
            endpoint=None,
            divider_detected=False,
            fill_ratio=0.0,
            features=("hook_bar_rect",),
        ),
        recorded_observation=True,
    )

    assert result.action_request.intent == ActionIntent.NONE


def test_episode_latch_survives_short_gap_but_uses_only_current_geometry() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, hook_disappearance_frames_required=2),
        initial_state=RuntimeState.HOOK,
    )
    gap = fsm.advance(
        _evidence(1), 0.2,
        _bundle(1, qualified=False, endpoint=None, divider_detected=False, fill_ratio=None),
        recorded_observation=True,
    )
    current = fsm.advance(
        _evidence(2), 0.4,
        _bundle(2, qualified=False, endpoint=1370.0),
        recorded_observation=True,
    )

    assert gap.action_request.intent == ActionIntent.NONE
    assert current.action_request.intent == ActionIntent.HOOK_ACTION
    assert current.action_request.payload["fill_endpoint_x"] == 1370.0


def test_episode_latch_clears_after_confirmed_bar_disappearance() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, hook_disappearance_frames_required=2),
        initial_state=RuntimeState.HOOK,
    )
    for frame in (1, 2):
        fsm.advance(
            _evidence(frame), frame * 0.2,
            _bundle(frame, qualified=False, endpoint=None, divider_detected=False, fill_ratio=None),
            recorded_observation=True,
        )
    stale_return = fsm.advance(
        _evidence(3), 0.6,
        _bundle(3, qualified=False, endpoint=1400.0),
        recorded_observation=True,
    )

    assert stale_return.action_request.intent == ActionIntent.NONE
    assert stale_return.transition_reason == "hook_episode_not_active"


def test_episode_latch_clears_on_timeout() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, hook_episode_timeout_sec=0.3),
        initial_state=RuntimeState.HOOK,
    )
    result = fsm.advance(
        _evidence(2), 0.4,
        _bundle(2, qualified=False, endpoint=1400.0),
        recorded_observation=True,
    )

    assert result.action_request.intent == ActionIntent.NONE
    assert result.transition_reason == "hook_episode_not_active"


def test_first_usable_post_cross_frame_does_not_add_artificial_delay() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.HOOK,
        initial_timestamp=77.6,
    )
    for frame in range(389, 396):
        held = fsm.advance(
            _evidence(frame), frame * 0.2,
            _bundle(frame, qualified=True, endpoint=None),
            recorded_observation=True,
        )
        assert held.action_request.intent == ActionIntent.NONE
    first_usable = fsm.advance(
        _evidence(396), 79.2,
        _bundle(396, qualified=True, endpoint=1411.0),
        recorded_observation=True,
    )

    assert first_usable.action_request.intent == ActionIntent.HOOK_ACTION

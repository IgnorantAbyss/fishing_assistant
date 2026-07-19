from pathlib import Path

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.collect_retry import CollectRetryConfig, CollectRetryController
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator
from src.fishing_v2.ports.action_sink import ActionExecutionResult
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationPolicy,
)
from src.fishing_v2.perception.observation_bundle import ObservationBundle


OPPORTUNITY = "cycle:1:COLLECT"


def _observe(
    controller: CollectRetryController,
    timestamp: float,
    *,
    visible: bool = True,
    observed: bool = True,
    opportunity: str = OPPORTUNITY,
):
    return controller.observe_panel(
        opportunity_id=opportunity,
        timestamp=timestamp,
        panel_observed=observed,
        panel_visible=visible,
        get_confidence=0.99,
        get_confirmation_frames=3,
    )


def _schedule(controller: CollectRetryController, timestamp: float):
    return controller.schedule_attempt(
        timestamp=timestamp,
        get_confidence=0.99,
        get_confirmation_frames=3,
    )


def _execution(
    action_id: str,
    timestamp: float,
    *,
    applied: bool = True,
    emitted: int = 2,
    expected: int = 2,
    partial: bool = False,
    reason: str | None = None,
) -> ActionExecutionResult:
    return ActionExecutionResult(
        action_id=action_id,
        intent_type="COLLECT",
        requested_at=timestamp,
        started_at=timestamp,
        completed_at=timestamp,
        success=applied,
        applied=applied,
        emitted_event_count=emitted,
        expected_event_count=expected,
        target_hwnd=10,
        foreground_hwnd=10,
        rejection_reason=reason,
        partial_execution=partial,
        os_input_emitted=applied and emitted == expected,
    )


def _complete(controller: CollectRetryController, timestamp: float):
    attempt, _ = _schedule(controller, timestamp)
    assert attempt is not None
    events = controller.record_execution(
        attempt, _execution(attempt.attempt_id, timestamp), timestamp=timestamp
    )
    return attempt, events


def test_collect_retry_config_matches_reviewed_live_policy() -> None:
    config = CollectRetryConfig()
    assert config.initial_settle_ms == 400
    assert config.retry_interval_ms == 350
    assert config.max_attempts == 12
    assert config.max_duration_seconds == 5.0
    assert config.disappearance_confirmation_frames == 2


def test_initial_settle_never_schedules_r() -> None:
    controller = CollectRetryController()
    _observe(controller, 10.0)
    attempt, _ = _schedule(controller, 10.399)
    assert attempt is None


def test_attempt_one_is_scheduled_after_initial_settle() -> None:
    controller = CollectRetryController()
    _observe(controller, 10.0)
    attempt, events = _schedule(controller, 10.4)
    assert attempt is not None
    assert attempt.attempt_number == 1
    assert attempt.attempt_id == "cycle:1:COLLECT:attempt:1"
    assert [event.event_type for event in events] == ["collect_attempt_scheduled"]


def test_attempt_two_requires_persistent_panel_and_complete_first_emission() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    _observe(controller, 0.75)
    attempt, _ = _schedule(controller, 0.75)
    assert attempt is not None
    assert attempt.attempt_number == 2


def test_retry_is_not_scheduled_before_interval() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    _observe(controller, 0.749)
    attempt, _ = _schedule(controller, 0.749)
    assert attempt is None


def test_attempt_ids_are_unique_within_opportunity() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    first, _ = _complete(controller, 0.4)
    _observe(controller, 0.75)
    second, _ = _complete(controller, 0.75)
    assert first.attempt_id != second.attempt_id
    assert first.opportunity_id == second.opportunity_id == OPPORTUNITY


def test_same_inflight_attempt_is_never_scheduled_twice() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    first, _ = _schedule(controller, 0.4)
    duplicate, _ = _schedule(controller, 0.8)
    assert first is not None
    assert duplicate is None


def test_stable_disappearance_stops_retry_and_acknowledges_collection() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    assert _observe(controller, 0.5, visible=False) == ()
    events = _observe(controller, 0.6, visible=False)
    assert [event.event_type for event in events] == ["collect_retry_succeeded"]
    assert controller.active is False
    assert controller.summary()["collect_completed_count"] == 1


def test_one_absent_frame_pauses_but_does_not_acknowledge() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    _observe(controller, 0.8, visible=False)
    attempt, _ = _schedule(controller, 0.8)
    assert attempt is None
    assert controller.visual_acknowledged is False


def test_max_attempts_exhausts_with_visual_ack_timeout() -> None:
    controller = CollectRetryController(CollectRetryConfig(max_attempts=2))
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    _observe(controller, 0.75)
    _complete(controller, 0.75)
    events = _observe(controller, 1.10)
    assert [event.event_type for event in events] == ["collect_retry_exhausted"]
    assert events[0].payload["outcome"] == "visual_ack_timeout"
    assert controller.summary()["collect_visual_timeout_count"] == 1


def test_max_duration_exhausts_even_before_attempt_limit() -> None:
    controller = CollectRetryController(
        CollectRetryConfig(max_attempts=99, max_duration_seconds=1.0)
    )
    _observe(controller, 0.0)
    events = _observe(controller, 1.0)
    assert events[0].payload["cancellation_reason"] == "max_duration_exceeded"


def test_focus_loss_rejection_cancels_retry() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    attempt, _ = _schedule(controller, 0.4)
    assert attempt is not None
    result = _execution(
        attempt.attempt_id, 0.4, applied=False, emitted=0,
        reason="foreground_window_mismatch",
    )
    events = controller.record_execution(attempt, result, timestamp=0.4)
    assert events[-1].payload["cancellation_reason"] == "foreground_window_mismatch"
    assert controller.active is False


def test_panic_rejection_cancels_retry() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    attempt, _ = _schedule(controller, 0.4)
    assert attempt is not None
    result = _execution(
        attempt.attempt_id, 0.4, applied=False, emitted=0,
        reason="panic_triggered",
    )
    events = controller.record_execution(attempt, result, timestamp=0.4)
    assert events[-1].payload["cancellation_reason"] == "panic_triggered"


def test_partial_sendinput_is_never_retried() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    attempt, _ = _schedule(controller, 0.4)
    assert attempt is not None
    result = _execution(
        attempt.attempt_id, 0.4, applied=False, emitted=1, partial=True,
        reason="sendinput_incomplete",
    )
    events = controller.record_execution(attempt, result, timestamp=0.4)
    assert events[-1].payload["cancellation_reason"] == "partial_os_input_not_retried"
    assert _schedule(controller, 2.0)[0] is None


def test_zero_emission_does_not_enter_high_speed_retry_loop() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    attempt, _ = _schedule(controller, 0.4)
    assert attempt is not None
    controller.record_execution(
        attempt,
        _execution(attempt.attempt_id, 0.4, applied=False, emitted=0),
        timestamp=0.4,
    )
    assert _schedule(controller, 0.401)[0] is None
    assert controller.active is False


def test_complete_emission_and_persistent_panel_permit_visual_retry() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    _, events = _complete(controller, 0.4)
    assert [event.event_type for event in events] == [
        "collect_attempt_emitted", "collect_attempt_waiting_ack",
    ]
    _observe(controller, 0.75)
    assert _schedule(controller, 0.75)[0] is not None


def test_result_banner_without_qualified_get_does_not_start_retry() -> None:
    controller = CollectRetryController()
    events = _observe(controller, 0.0, visible=False, observed=False)
    assert events == ()
    assert controller.active is False


def test_next_get_episode_restarts_at_attempt_one_without_leakage() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    _observe(controller, 0.5, visible=False)
    _observe(controller, 0.6, visible=False)
    _observe(controller, 2.0, opportunity="cycle:2:COLLECT")
    attempt, _ = _schedule(controller, 2.4)
    assert attempt is not None
    assert attempt.attempt_id == "cycle:2:COLLECT:attempt:1"


def test_non_collect_intents_keep_original_one_shot_deduplication() -> None:
    tracker = WouldFireDeduplicator()
    request = ActionRequest(ActionIntent.HOOK_ACTION, 0.99, "crossed")
    values = dict(
        safety_reason="action_emission_disabled",
        frame_index=1,
        timestamp=1.0,
        runtime_state="HOOK",
        prompt_evidence={},
        specialized_evidence={},
    )
    assert tracker.observe(request, **values) is not None
    assert tracker.observe(request, **values) is None


def test_collect_pending_keeps_get_detector_armed_for_disappearance_confirmation() -> None:
    activation = DetectorActivationPolicy().evaluate(
        RuntimeState.COLLECT_PENDING, ObservationBundle(1, 1.0),
        recorded_observation=True,
    )
    assert activation.get == DetectorActivationMode.ARMED


def test_collect_retry_module_contains_no_real_input_backend() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "fishing_v2" / "live" / "collect_retry.py"
    ).read_text(encoding="utf-8")
    assert "SendInput(" not in source
    assert "keybd_event" not in source

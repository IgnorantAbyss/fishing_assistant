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
    assert attempt.attempt_id == "get_episode:1:COLLECT:attempt:1"
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
    assert first.opportunity_id == second.opportunity_id == "get_episode:1:COLLECT"


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
    assert [event.event_type for event in events] == [
        "collect_retry_succeeded", "get_episode_completed",
    ]
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
    assert attempt.attempt_id == "get_episode:2:COLLECT:attempt:1"


def test_runtime_cycle_change_does_not_rebuild_persistent_get_opportunity() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0, opportunity="cycle:1:COLLECT")
    first, _ = _complete(controller, 0.4)
    events = _observe(controller, 0.5, opportunity="cycle:2:COLLECT")
    assert events == ()
    _observe(controller, 0.75, opportunity="cycle:3:COLLECT")
    second, _ = _complete(controller, 0.75)
    assert first.opportunity_id == second.opportunity_id == "get_episode:1:COLLECT"
    assert second.attempt_number == 2
    assert controller.summary()["physical_get_episode_count"] == 1
    assert controller.summary()["collect_opportunity_count"] == 1


def test_terminal_get_episode_stays_terminal_across_recovery_cycles() -> None:
    controller = CollectRetryController(CollectRetryConfig(max_attempts=1))
    _observe(controller, 0.0, opportunity="cycle:1:COLLECT")
    _complete(controller, 0.4)
    _observe(controller, 0.75, opportunity="cycle:2:COLLECT")
    assert controller.active is False
    for cycle in range(3, 7):
        events = _observe(
            controller, float(cycle), opportunity=f"cycle:{cycle}:COLLECT"
        )
        assert all(event.event_type != "get_episode_started" for event in events)
        assert _schedule(controller, float(cycle))[0] is None
    summary = controller.summary()
    assert summary["physical_get_episode_count"] == 1
    assert summary["collect_attempt_counts_by_get_episode"] == {
        "get_episode:1": 1
    }
    assert summary["collect_terminal_episode_count"] == 1


def test_same_physical_get_panel_never_exceeds_twelve_inputs_across_cycles() -> None:
    controller = CollectRetryController()
    _observe(controller, 0.0, opportunity="cycle:1:COLLECT")
    attempts = []
    for number in range(1, 13):
        timestamp = 0.401 + (number - 1) * 0.351
        _observe(
            controller, timestamp,
            opportunity=f"cycle:{number}:COLLECT",
        )
        attempt, _ = _complete(controller, timestamp)
        attempts.append(attempt)
    events = _observe(controller, 4.7, opportunity="cycle:99:COLLECT")
    assert [item.event_type for item in events] == ["collect_retry_exhausted"]
    assert [item.attempt_number for item in attempts] == list(range(1, 13))
    assert {item.opportunity_id for item in attempts} == {
        "get_episode:1:COLLECT"
    }
    for timestamp in (5.0, 10.0, 30.0):
        _observe(controller, timestamp, opportunity="cycle:100:COLLECT")
        assert _schedule(controller, timestamp)[0] is None
    assert controller.summary()["collect_attempt_counts_by_get_episode"] == {
        "get_episode:1": 12
    }


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


def test_distinct_physical_get_attempts_have_distinct_deduplication_identity() -> None:
    controller = CollectRetryController()
    tracker = WouldFireDeduplicator()
    request = ActionRequest(ActionIntent.COLLECT, 0.99, "get_panel_present")
    values = dict(
        safety_reason="action_emission_disabled",
        frame_index=1,
        timestamp=1.0,
        runtime_state="GET",
        prompt_evidence={},
        specialized_evidence={},
        count_raw=False,
    )

    _observe(controller, 0.0)
    first, _ = _schedule(controller, 0.4)
    assert first is not None
    assert tracker.observe(
        request, identity_suffix=first.deduplication_identity, **values
    ) is not None
    controller.record_execution(
        first, _execution(first.attempt_id, 0.4), timestamp=0.4
    )
    _observe(controller, 0.5, visible=False)
    _observe(controller, 0.6, visible=False)

    _observe(controller, 1.0, opportunity="cycle:1:COLLECT")
    second, _ = _schedule(controller, 1.4)
    assert second is not None
    assert second.attempt_number == first.attempt_number == 1
    assert second.deduplication_identity != first.deduplication_identity
    assert tracker.observe(
        request,
        identity_suffix=second.deduplication_identity,
        timestamp=1.4,
        **{key: value for key, value in values.items() if key != "timestamp"},
    ) is not None


def test_zero_emission_terminal_sync_recovery_rearms_once_with_new_identity() -> None:
    controller = CollectRetryController(
        CollectRetryConfig(max_duration_seconds=1.0)
    )
    _observe(controller, 0.0)
    first, _ = _schedule(controller, 0.4)
    assert first is not None
    events = _observe(controller, 1.01)
    assert [event.event_type for event in events] == ["collect_retry_exhausted"]

    serviceable, recovery_events = controller.prepare_sync_recovery(
        opportunity_id="cycle:1:COLLECT",
        timestamp=1.4,
        panel_observed=True,
        panel_visible=True,
        get_confidence=0.99,
        get_confirmation_frames=6,
    )
    assert serviceable is True
    assert [event.event_type for event in recovery_events] == [
        "collect_rearmed_after_sync_recovery"
    ]
    payload = controller.lifecycle_payload(1.4)
    assert payload["physical_get_episode_id"] == "get_episode:1"
    assert payload["collect_opportunity_id"] == (
        "get_episode:1:COLLECT:recovery:1"
    )
    assert payload["recovery_generation"] == 1
    assert payload["serviceable"] is True

    recovered, _ = _schedule(controller, 1.8)
    assert recovered is not None
    assert recovered.deduplication_identity != first.deduplication_identity

    _observe(controller, 2.41)
    assert controller.episode_terminal is True
    serviceable, recovery_events = controller.prepare_sync_recovery(
        opportunity_id="cycle:1:COLLECT",
        timestamp=2.5,
        panel_observed=True,
        panel_visible=True,
        get_confidence=0.99,
        get_confirmation_frames=6,
    )
    assert serviceable is False
    assert recovery_events == ()
    assert controller.lifecycle_payload(2.5)["service_block_reason"] == (
        "zero_emission_sync_recovery_already_used"
    )


def test_successful_emission_terminal_is_not_rearmed_by_sync_recovery() -> None:
    controller = CollectRetryController(
        CollectRetryConfig(max_duration_seconds=1.0)
    )
    _observe(controller, 0.0)
    _complete(controller, 0.4)
    _observe(controller, 1.01)

    serviceable, events = controller.prepare_sync_recovery(
        opportunity_id="cycle:1:COLLECT",
        timestamp=1.4,
        panel_observed=True,
        panel_visible=True,
        get_confidence=0.99,
        get_confirmation_frames=6,
    )
    assert serviceable is False
    assert events == ()
    payload = controller.lifecycle_payload(1.4)
    assert payload["actual_emission_count"] == 1
    assert payload["service_block_reason"] == (
        "terminal_after_actual_os_emission"
    )


def test_collect_sync_recovery_is_bounded_across_one_thousand_observations() -> None:
    controller = CollectRetryController(
        CollectRetryConfig(max_duration_seconds=1.0)
    )
    _observe(controller, 0.0)
    first, _ = _schedule(controller, 0.4)
    assert first is not None
    _observe(controller, 1.01)

    serviceable, events = controller.prepare_sync_recovery(
        opportunity_id="cycle:1:COLLECT",
        timestamp=1.4,
        panel_observed=True,
        panel_visible=True,
        get_confidence=0.99,
        get_confirmation_frames=6,
    )
    assert serviceable is True
    assert sum(
        event.event_type == "collect_rearmed_after_sync_recovery"
        for event in events
    ) == 1
    recovered, _ = _schedule(controller, 1.8)
    assert recovered is not None
    _observe(controller, 2.41)

    for index in range(1000):
        serviceable, events = controller.prepare_sync_recovery(
            opportunity_id="cycle:1:COLLECT",
            timestamp=2.5 + index * 0.001,
            panel_observed=True,
            panel_visible=True,
            get_confidence=0.99,
            get_confirmation_frames=6,
        )
        assert serviceable is False
        assert events == ()
    assert controller.summary()["collect_opportunity_count"] == 2


def test_collect_retry_module_contains_no_real_input_backend() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "fishing_v2" / "live" / "collect_retry.py"
    ).read_text(encoding="utf-8")
    assert "SendInput(" not in source
    assert "keybd_event" not in source

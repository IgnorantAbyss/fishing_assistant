from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.cast_opportunity import (
    CastOpportunityConfig,
    CastOpportunityController,
)
from src.fishing_v2.ports.action_sink import ActionExecutionResult


def _execution(action_id: str, timestamp: float, *, applied: bool = True):
    return ActionExecutionResult(
        action_id=action_id,
        intent_type="CAST",
        requested_at=timestamp,
        started_at=timestamp,
        completed_at=timestamp,
        success=applied,
        applied=applied,
        emitted_event_count=2 if applied else 0,
        expected_event_count=2,
        target_hwnd=10,
        foreground_hwnd=10,
        rejection_reason=None if applied else "foreground_window_mismatch",
        os_input_emitted=applied,
    )


def _schedule(controller: CastOpportunityController, timestamp: float = 0.0):
    return controller.schedule(
        timestamp=timestamp,
        idle_cast_prompt=True,
        get_observed_absent=True,
        result_banner_absent=True,
    )


def test_cast_requires_idle_prompt_get_absence_and_result_banner_absence() -> None:
    controller = CastOpportunityController()
    for values in ((False, True, True), (True, False, True), (True, True, False)):
        attempt, events = controller.schedule(
            timestamp=0.0,
            idle_cast_prompt=values[0],
            get_observed_absent=values[1],
            result_banner_absent=values[2],
        )
        assert attempt is None
        assert events == ()


def test_cast_is_one_shot_until_waiting_visual_acknowledgement() -> None:
    controller = CastOpportunityController()
    attempt, events = _schedule(controller)
    assert attempt is not None
    assert attempt.action_id == "cast_opportunity:1:CAST"
    assert [item.event_type for item in events] == ["cast_opportunity_started"]
    controller.record_execution(attempt, _execution(attempt.action_id, 0.0), timestamp=0.0)
    assert _schedule(controller, 0.1)[0] is None
    assert controller.observe(
        timestamp=0.5,
        runtime_state=RuntimeState.CAST_PENDING,
        prompt_kind=PromptObservationKind.UNKNOWN,
    ) == ()
    acknowledged = controller.observe(
        timestamp=0.6,
        runtime_state=RuntimeState.WAITING,
        prompt_kind=PromptObservationKind.WAITING_IN_PROGRESS,
    )
    assert [item.event_type for item in acknowledged] == ["cast_visual_acknowledged"]
    next_attempt, _ = _schedule(controller, 2.0)
    assert next_attempt is not None
    assert next_attempt.action_id == "cast_opportunity:2:CAST"


def test_cast_timeout_is_terminal_and_never_retries_across_sync_cycles() -> None:
    controller = CastOpportunityController(CastOpportunityConfig(
        visual_ack_timeout_seconds=4.0
    ))
    attempt, _ = _schedule(controller)
    assert attempt is not None
    controller.record_execution(attempt, _execution(attempt.action_id, 0.0), timestamp=0.0)
    timeout = controller.observe(
        timestamp=4.0,
        runtime_state=RuntimeState.SYNC_REQUIRED,
        prompt_kind=PromptObservationKind.IDLE_CAST,
    )
    assert [item.event_type for item in timeout] == ["cast_visual_timeout"]
    assert timeout[0].payload["retry_scheduled"] is False
    for timestamp in (4.1, 8.0, 20.0):
        assert _schedule(controller, timestamp)[0] is None
    assert controller.summary() == {
        "cast_opportunity_count": 1,
        "cast_attempt_count": 1,
        "cast_visual_acknowledged_count": 0,
        "cast_timeout_count": 1,
    }


def test_rejected_or_partial_cast_is_never_retried() -> None:
    controller = CastOpportunityController()
    attempt, _ = _schedule(controller)
    assert attempt is not None
    controller.record_execution(
        attempt, _execution(attempt.action_id, 0.0, applied=False), timestamp=0.0
    )
    assert _schedule(controller, 1.0)[0] is None
    assert controller.summary()["cast_attempt_count"] == 1

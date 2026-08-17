from __future__ import annotations

import pytest

from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.runtime.scheduling import (
    QuietWaitingCaptureScheduler,
)


def _quiet_kwargs() -> dict[str, bool]:
    return {
        "ready_candidate_active": False,
        "recovery_escalation": False,
        "critical_detector_active": False,
    }


def _simulate_waiting(duration_seconds: float) -> tuple[int, dict[str, float | int]]:
    scheduler = QuietWaitingCaptureScheduler(5.0)
    captures = 0
    control_ticks = int(round(duration_seconds * 25.0))
    for tick in range(control_ticks):
        timestamp = tick / 25.0
        scheduler.observe_control_tick(timestamp, RuntimeState.WAITING)
        if scheduler.should_capture(
            timestamp, RuntimeState.WAITING, **_quiet_kwargs()
        ):
            captures += 1
            scheduler.record_capture(
                timestamp,
                state_at_capture=RuntimeState.WAITING,
                state_after_processing=RuntimeState.WAITING,
                **_quiet_kwargs(),
            )
    scheduler.finish(duration_seconds, RuntimeState.WAITING)
    return captures, scheduler.summary()


def test_quiet_waiting_uses_five_second_capture_interval() -> None:
    scheduler = QuietWaitingCaptureScheduler(5.0)
    assert scheduler.should_capture(
        0.0, RuntimeState.WAITING, **_quiet_kwargs()
    )
    scheduler.record_capture(
        0.0,
        state_at_capture=RuntimeState.WAITING,
        state_after_processing=RuntimeState.WAITING,
        **_quiet_kwargs(),
    )
    assert scheduler.next_capture_due == pytest.approx(5.0)
    assert not scheduler.should_capture(
        4.999, RuntimeState.WAITING, **_quiet_kwargs()
    )
    assert scheduler.should_capture(
        5.0, RuntimeState.WAITING, **_quiet_kwargs()
    )


def test_long_waiting_measurement_is_bounded_without_deadline_drift() -> None:
    minute_captures, _ = _simulate_waiting(60.0)
    captures, summary = _simulate_waiting(600.0)
    legacy_captures = int(600.0 * 25.0)

    assert minute_captures == 12
    assert captures == 120
    assert legacy_captures == 15_000
    assert summary["control_loop_iterations"] == 15_000
    assert summary["waiting_effective_capture_fps"] == pytest.approx(0.2)


def test_ready_candidate_immediately_exits_quiet_waiting_cadence() -> None:
    scheduler = QuietWaitingCaptureScheduler(5.0)
    scheduler.should_capture(0.0, RuntimeState.WAITING, **_quiet_kwargs())
    scheduler.record_capture(
        0.0,
        state_at_capture=RuntimeState.WAITING,
        state_after_processing=RuntimeState.WAITING,
        **_quiet_kwargs(),
    )
    assert scheduler.should_capture(
        5.0, RuntimeState.WAITING, **_quiet_kwargs()
    )
    scheduler.record_capture(
        5.0,
        state_at_capture=RuntimeState.WAITING,
        state_after_processing=RuntimeState.WAITING,
        ready_candidate_active=True,
        recovery_escalation=False,
        critical_detector_active=False,
    )
    assert scheduler.next_capture_due is None
    assert scheduler.should_capture(
        5.04,
        RuntimeState.WAITING,
        ready_candidate_active=True,
        recovery_escalation=False,
        critical_detector_active=False,
    )


def test_waiting_transition_arms_fresh_deadline_for_each_episode() -> None:
    scheduler = QuietWaitingCaptureScheduler(5.0)
    scheduler.record_capture(
        10.0,
        state_at_capture=RuntimeState.CAST_PENDING,
        state_after_processing=RuntimeState.WAITING,
        **_quiet_kwargs(),
    )
    assert scheduler.next_capture_due == pytest.approx(15.0)
    assert not scheduler.should_capture(
        10.04, RuntimeState.WAITING, **_quiet_kwargs()
    )

    scheduler.record_capture(
        15.0,
        state_at_capture=RuntimeState.WAITING,
        state_after_processing=RuntimeState.READY,
        **_quiet_kwargs(),
    )
    assert scheduler.next_capture_due is None
    scheduler.record_capture(
        20.0,
        state_at_capture=RuntimeState.CAST_PENDING,
        state_after_processing=RuntimeState.WAITING,
        **_quiet_kwargs(),
    )
    assert scheduler.next_capture_due == pytest.approx(25.0)


@pytest.mark.parametrize(
    "state",
    [
        RuntimeState.READY,
        RuntimeState.HOOK_PENDING,
        RuntimeState.HOOK,
        RuntimeState.PRESS,
        RuntimeState.GET,
        RuntimeState.RESULT_PENDING,
        RuntimeState.IDLE,
        RuntimeState.CAST_PENDING,
        RuntimeState.COLLECT_PENDING,
        RuntimeState.SYNC_REQUIRED,
    ],
)
def test_non_waiting_states_never_use_quiet_capture_gate(
    state: RuntimeState,
) -> None:
    scheduler = QuietWaitingCaptureScheduler(5.0)
    assert scheduler.should_capture(0.0, state, **_quiet_kwargs())
    scheduler.record_capture(
        0.0,
        state_at_capture=state,
        state_after_processing=state,
        **_quiet_kwargs(),
    )
    assert scheduler.next_capture_due is None


def test_recovery_or_critical_activity_disables_quiet_capture_gate() -> None:
    scheduler = QuietWaitingCaptureScheduler(5.0)
    for recovery, critical in ((True, False), (False, True)):
        assert scheduler.should_capture(
            0.0,
            RuntimeState.WAITING,
            ready_candidate_active=False,
            recovery_escalation=recovery,
            critical_detector_active=critical,
        )

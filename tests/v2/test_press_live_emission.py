from __future__ import annotations

import random
import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.live.press_live_emission import (
    PressLiveEmissionConfig,
    PressLiveEmissionTracker,
    pending_press_cancellation_reason,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.ports.action_sink import ActionExecutionResult
from src.fishing_v2.runtime.press_action_contract import (
    validate_frozen_press_payload,
)


def _execution(
    *,
    action_id: str,
    applied: bool,
    partial: bool = False,
    attempted: int,
    completed: int,
    total: int,
) -> ActionExecutionResult:
    return ActionExecutionResult(
        action_id=action_id,
        intent_type=ActionIntent.PRESS_SEQUENCE.value,
        requested_at=1.0,
        started_at=1.01,
        completed_at=1.2,
        success=applied,
        applied=applied,
        emitted_event_count=completed * 2,
        expected_event_count=total * 2,
        target_hwnd=4242,
        foreground_hwnd=4242,
        partial_execution=partial,
        os_input_emitted=applied,
        attempted_count=attempted,
        completed_key_count=completed,
        total_key_count=total,
    )


def test_press_emission_is_once_per_episode_but_resets_for_next_episode() -> None:
    tracker = PressLiveEmissionTracker()
    first, _ = tracker.begin_attempt(
        episode_index=1,
        timestamp=1.0,
        sequence=tuple("WWAD"),
    )
    repeated, blocked = tracker.begin_attempt(
        episode_index=1,
        timestamp=1.1,
        sequence=tuple("WWAD"),
    )
    second, _ = tracker.begin_attempt(
        episode_index=2,
        timestamp=2.0,
        sequence=tuple("ASDWAS"),
    )
    assert first is True
    assert repeated is False
    assert blocked[0].payload["reason"] == (
        "press_episode_emission_already_attempted"
    )
    assert second is True


def test_press_schedule_samples_bounded_reproducible_pacing() -> None:
    config = PressLiveEmissionConfig(
        initial_delay_min_ms=300,
        initial_delay_max_ms=500,
        inter_key_gap_min_ms=30,
        inter_key_gap_max_ms=80,
        key_hold_ms=40,
    )
    first = PressLiveEmissionTracker(config, rng=random.Random(73))
    second = PressLiveEmissionTracker(config, rng=random.Random(73))
    scheduled_a, events_a = first.schedule(
        episode_index=1,
        timestamp=10.0,
        sequence=tuple("WWAD"),
        slot_capacity=8,
    )
    scheduled_b, _ = second.schedule(
        episode_index=1,
        timestamp=10.0,
        sequence=tuple("WWAD"),
        slot_capacity=8,
    )
    assert scheduled_a is not None and scheduled_b is not None
    assert scheduled_a.timing == scheduled_b.timing
    assert 300 <= scheduled_a.timing.sampled_initial_delay_ms <= 500
    assert scheduled_a.timing.key_hold_ms == (40, 40, 40, 40)
    assert len(scheduled_a.timing.inter_key_gap_ms) == 3
    assert all(
        30 <= gap <= 80
        for gap in scheduled_a.timing.inter_key_gap_ms
    )
    assert scheduled_a.deadline == pytest.approx(
        10.0 + scheduled_a.timing.sampled_initial_delay_ms / 1000.0
    )
    assert events_a[0].payload["planned_total_duration_ms"] == (
        scheduled_a.timing.planned_total_duration_ms
    )


def test_press_schedule_is_nonblocking_and_cancelled_episode_is_not_retried() -> None:
    tracker = PressLiveEmissionTracker(
        PressLiveEmissionConfig(
            initial_delay_min_ms=300,
            initial_delay_max_ms=300,
        )
    )
    scheduled, _ = tracker.schedule(
        episode_index=4,
        timestamp=2.0,
        sequence=tuple("WAS"),
        slot_capacity=8,
    )
    assert scheduled is not None
    assert tracker.due(2.299) is False
    assert tracker.due(2.300) is True
    cancelled = tracker.cancel_pending(
        timestamp=2.1,
        reason="press_panel_disappeared",
    )
    assert cancelled[0].event_type == "press_emission_cancelled"
    repeated, blocked = tracker.schedule(
        episode_index=4,
        timestamp=2.2,
        sequence=tuple("WAS"),
        slot_capacity=8,
    )
    assert repeated is None
    assert blocked[0].payload["reason"] == (
        "press_episode_opportunity_already_reserved"
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"runtime_state": RuntimeState.RESULT_PENDING}, "runtime_left_press"),
        ({"panel_disappeared": True}, "press_panel_disappeared"),
        ({"foreground": False}, "foreground_not_confirmed"),
        ({"panic_triggered": True}, "panic_triggered"),
    ],
)
def test_initial_delay_context_loss_cancels_before_any_attempt(
    overrides: dict,
    reason: str,
) -> None:
    tracker = PressLiveEmissionTracker(
        PressLiveEmissionConfig(
            initial_delay_min_ms=300,
            initial_delay_max_ms=300,
        )
    )
    pending, _ = tracker.schedule(
        episode_index=7,
        timestamp=1.0,
        sequence=tuple("WAS"),
        slot_capacity=8,
    )
    assert pending is not None
    context = {
        "runtime_state": RuntimeState.PRESS,
        "active_episode": True,
        "episode_index": 7,
        "frozen_sequence": tuple("WAS"),
        "panel_disappeared": False,
        "foreground": True,
        "panic_triggered": False,
    }
    context.update(overrides)
    assert pending_press_cancellation_reason(
        pending,
        **context,
    ) == reason
    tracker.cancel_pending(timestamp=1.1, reason=reason)
    assert tracker.summary()["press_live_emission_attempted_count"] == 0


def test_partial_press_emission_is_terminal_and_not_waiting_for_ack() -> None:
    tracker = PressLiveEmissionTracker()
    tracker.begin_attempt(
        episode_index=1,
        timestamp=1.0,
        sequence=tuple("WASD"),
    )
    events = tracker.record_execution(
        episode_index=1,
        timestamp=1.2,
        execution=_execution(
            action_id="press:1",
            applied=False,
            partial=True,
            attempted=3,
            completed=2,
            total=4,
        ),
    )
    assert events[0].event_type == "press_emission_partial"
    assert events[0].payload["completed_key_count"] == 2
    summary = tracker.summary()
    assert summary["press_live_emission_partial_count"] == 1
    assert summary["press_live_awaiting_visual_ack"] is False


def test_complete_press_waits_for_panel_disappearance_or_times_out() -> None:
    tracker = PressLiveEmissionTracker(
        PressLiveEmissionConfig(visual_ack_timeout_seconds=1.0)
    )
    tracker.begin_attempt(
        episode_index=1,
        timestamp=1.0,
        sequence=tuple("WASD"),
    )
    tracker.record_execution(
        episode_index=1,
        timestamp=1.2,
        execution=_execution(
            action_id="press:1",
            applied=True,
            attempted=4,
            completed=4,
            total=4,
        ),
    )
    assert tracker.observe_panel(
        timestamp=1.5,
        panel_observed=True,
        panel_disappeared=False,
    ) == ()
    acknowledged = tracker.observe_panel(
        timestamp=1.6,
        panel_observed=True,
        panel_disappeared=True,
    )
    assert acknowledged[0].event_type == (
        "press_visual_acknowledged"
    )

    tracker.begin_attempt(
        episode_index=2,
        timestamp=2.0,
        sequence=tuple("WW"),
    )
    tracker.record_execution(
        episode_index=2,
        timestamp=2.1,
        execution=_execution(
            action_id="press:2",
            applied=True,
            attempted=2,
            completed=2,
            total=2,
        ),
    )
    timed_out = tracker.observe_panel(
        timestamp=3.2,
        panel_observed=False,
        panel_disappeared=False,
    )
    assert timed_out[0].event_type == "press_visual_ack_timeout"
    assert timed_out[0].payload["terminal_outcome"] == (
        "visual_ack_timeout_not_retried"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "sequence": (),
            "slot_capacity": 8,
            "active_press_episode": True,
            "panel_confirmed": True,
            "frozen_by_consensus": True,
        },
        {
            "sequence": ("W", "X"),
            "slot_capacity": 8,
            "active_press_episode": True,
            "panel_confirmed": True,
            "frozen_by_consensus": True,
        },
        {
            "sequence": ("W", "A", "S", "D"),
            "slot_capacity": 3,
            "active_press_episode": True,
            "panel_confirmed": True,
            "frozen_by_consensus": True,
        },
    ],
)
def test_invalid_frozen_payload_is_rejected_without_transformation(
    payload: dict,
) -> None:
    with pytest.raises(ValueError):
        validate_frozen_press_payload(payload)


def test_frozen_payload_preserves_order_and_repetition() -> None:
    sequence, capacity = validate_frozen_press_payload({
        "sequence": ("W", "W", "A", "D"),
        "slot_capacity": 8,
        "active_press_episode": True,
        "panel_confirmed": True,
        "frozen_by_consensus": True,
    })
    assert sequence == ("W", "W", "A", "D")
    assert capacity == 8

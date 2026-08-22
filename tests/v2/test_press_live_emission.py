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
from src.fishing_v2.fusion.observation_fusion import StateEvidence
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


def test_production_v3_shortens_only_freeze_to_first_emission_deadline() -> None:
    freeze_at = 10.0
    legacy = PressLiveEmissionTracker(PressLiveEmissionConfig(
        initial_delay_min_ms=300,
        initial_delay_max_ms=300,
        inter_key_gap_min_ms=90,
        inter_key_gap_max_ms=90,
        key_hold_ms=40,
    ))
    production_v3 = PressLiveEmissionTracker(PressLiveEmissionConfig(
        initial_delay_min_ms=150,
        initial_delay_max_ms=150,
        inter_key_gap_min_ms=90,
        inter_key_gap_max_ms=90,
        key_hold_ms=40,
    ))

    legacy_pending, _ = legacy.schedule(
        episode_index=1,
        timestamp=freeze_at,
        sequence=tuple("ASDWDAS"),
        slot_capacity=8,
    )
    v3_pending, v3_events = production_v3.schedule(
        episode_index=1,
        timestamp=freeze_at,
        sequence=tuple("ASDWDAS"),
        slot_capacity=8,
    )

    assert legacy_pending is not None and v3_pending is not None
    assert legacy_pending.scheduled_at == v3_pending.scheduled_at == freeze_at
    assert legacy_pending.deadline == pytest.approx(10.300)
    assert v3_pending.deadline == pytest.approx(10.150)
    assert production_v3.due(10.149) is False
    assert production_v3.due(10.150) is True
    started, _ = production_v3.begin_scheduled_attempt(timestamp=10.150)
    assert started is not None
    assert started.sequence == tuple("ASDWDAS")
    assert started.timing.key_hold_ms == (40,) * 7
    assert started.timing.inter_key_gap_ms == (90,) * 6
    assert v3_events[0].payload["timestamp"] == freeze_at
    assert v3_events[0].payload["deadline"] == pytest.approx(10.150)

    repeated, blocked = production_v3.schedule(
        episode_index=1,
        timestamp=10.151,
        sequence=tuple("ASDWDAS"),
        slot_capacity=8,
    )
    assert repeated is None
    assert blocked[0].payload["reason"] == (
        "press_episode_opportunity_already_reserved"
    )


class ScriptedTimingRng:
    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)
        self.calls: list[tuple[int, int]] = []

    def randint(self, lower: int, upper: int) -> int:
        self.calls.append((lower, upper))
        value = next(self.values)
        assert lower <= value <= upper
        return value


def test_each_press_key_gap_is_sampled_independently_and_plan_is_frozen() -> None:
    rng = ScriptedTimingRng([437, 91, 135, 169])
    tracker = PressLiveEmissionTracker(
        PressLiveEmissionConfig(),
        rng=rng,
    )

    scheduled, events = tracker.schedule(
        episode_index=8,
        timestamp=10.0,
        sequence=tuple("WASD"),
        slot_capacity=8,
    )

    assert scheduled is not None
    assert rng.calls == [
        (300, 500),
        (90, 170),
        (90, 170),
        (90, 170),
    ]
    assert scheduled.timing.sampled_initial_delay_ms == 437
    assert scheduled.timing.key_hold_ms == (40, 40, 40, 40)
    assert scheduled.timing.inter_key_gap_ms == (91, 135, 169)
    assert scheduled.timing.planned_total_duration_ms == 992
    assert events[0].payload["inter_key_gap_ms"] == [91, 135, 169]
    assert scheduled.timing.console_schedule(tuple("WASD")) == (
        "PRESS scheduled: sequence=WASD initial_delay_ms=437 "
        "hold_ms=[40,40,40,40] gap_ms=[91,135,169] "
        "planned_total_duration_ms=992"
    )

    frozen_plan = scheduled.timing
    started, _ = tracker.begin_scheduled_attempt(timestamp=10.437)
    assert started is not None
    assert started.timing is frozen_plan
    assert rng.calls == [
        (300, 500),
        (90, 170),
        (90, 170),
        (90, 170),
    ]


def test_equal_press_timing_bounds_produce_fixed_values() -> None:
    tracker = PressLiveEmissionTracker(PressLiveEmissionConfig(
        initial_delay_min_ms=350,
        initial_delay_max_ms=350,
        inter_key_gap_min_ms=45,
        inter_key_gap_max_ms=45,
    ))

    scheduled, _ = tracker.schedule(
        episode_index=9,
        timestamp=1.0,
        sequence=tuple("WASD"),
        slot_capacity=8,
    )

    assert scheduled is not None
    assert scheduled.timing.sampled_initial_delay_ms == 350
    assert scheduled.timing.inter_key_gap_ms == (45, 45, 45)


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
        ({"active_episode": False}, "press_episode_changed"),
        ({"episode_index": 8}, "press_episode_changed"),
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


def test_default_press_gap_range_is_90_to_170_per_adjacent_key() -> None:
    tracker = PressLiveEmissionTracker(
        rng=random.Random(20260801),
    )
    scheduled, _ = tracker.schedule(
        episode_index=11,
        timestamp=1.0,
        sequence=tuple("WSSAA"),
        slot_capacity=8,
    )
    assert scheduled is not None
    assert len(scheduled.timing.inter_key_gap_ms) == 4
    assert all(
        90 <= value <= 170
        for value in scheduled.timing.inter_key_gap_ms
    )
    assert scheduled.sequence == tuple("WSSAA")


@pytest.mark.parametrize(
    ("current_candidate", "sequence_ready"),
    [
        (tuple("DASWW"), True),
        (tuple("WSSAA"), False),
        ((), False),
    ],
)
def test_frozen_press_snapshot_survives_recognition_drift_until_deadline(
    current_candidate: tuple[str, ...],
    sequence_ready: bool,
) -> None:
    frozen_evidence = StateEvidence(
        {RuntimeState.PRESS: 0.99},
        ("press_consensus:0.990",),
        (),
        RuntimeState.PRESS,
        0.99,
        "press_sequence_frozen",
        10,
        1.0,
    )
    tracker = PressLiveEmissionTracker(PressLiveEmissionConfig(
        initial_delay_min_ms=400,
        initial_delay_max_ms=400,
    ))
    pending, _ = tracker.schedule(
        episode_index=12,
        timestamp=1.0,
        sequence=tuple("WSSAA"),
        slot_capacity=8,
        frozen_evidence=frozen_evidence,
    )
    assert pending is not None

    # Post-freeze recognition is deliberately different, not-ready, or empty.
    # It is not an input to pending eligibility.
    post_freeze_recognition = {
        "sequence_candidate": current_candidate,
        "sequence_ready": sequence_ready,
    }
    assert post_freeze_recognition != {
        "sequence_candidate": pending.sequence,
        "sequence_ready": True,
    }
    assert pending.sequence == tuple("WSSAA")
    assert pending_press_cancellation_reason(
        pending,
        runtime_state=RuntimeState.PRESS,
        active_episode=True,
        episode_index=12,
        panel_disappeared=False,
        foreground=True,
        panic_triggered=False,
    ) is None
    assert tracker.due(1.399) is False
    assert tracker.due(1.400) is True
    started, _ = tracker.begin_scheduled_attempt(timestamp=1.400)
    assert started is not None
    assert started.sequence == tuple("WSSAA")
    assert started.slot_capacity == 8
    assert started.timing is pending.timing
    assert started.frozen_evidence is frozen_evidence


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

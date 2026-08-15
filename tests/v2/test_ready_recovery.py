from __future__ import annotations

from src.fishing_v2.domain.observations import (
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.ready_recovery import (
    ReadyRecoveryConfig,
    ReadyRecoveryTracker,
)


def _prompt(
    frame_index: int,
    timestamp: float,
    *,
    kind: PromptObservationKind = PromptObservationKind.READY_BITE,
    confidence: float = 0.99,
) -> PromptObservation:
    return PromptObservation(
        kind,
        confidence,
        {kind.value: confidence},
        "ready-recovery-test",
        frame_index,
        timestamp,
    )


def _observe(
    tracker: ReadyRecoveryTracker,
    frame_index: int,
    timestamp: float,
    *,
    foreground: bool = True,
    state: RuntimeState = RuntimeState.READY,
    kind: PromptObservationKind = PromptObservationKind.READY_BITE,
    confidence: float = 0.99,
    conflict: bool = False,
    panic: bool = False,
):
    return tracker.observe(
        frame_index=frame_index,
        timestamp=timestamp,
        runtime_state=state,
        prompt=_prompt(
            frame_index,
            timestamp,
            kind=kind,
            confidence=confidence,
        ),
        foreground=foreground,
        target_valid=True,
        conflicting_evidence=conflict,
        panic_triggered=panic,
        action_emission_in_progress=False,
    )


def test_foreground_restore_uses_only_new_stable_ready_frames() -> None:
    tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(2, 0.8))

    lost = _observe(tracker, 10, 1.0, foreground=False)
    restored = _observe(tracker, 11, 1.1, foreground=True)
    first_new = _observe(tracker, 12, 1.15)
    certified = _observe(tracker, 13, 1.2)

    assert [item.event_type for item in lost.events] == ["foreground_lost"]
    assert [item.event_type for item in restored.events] == [
        "foreground_restored"
    ]
    assert restored.certificate is None
    assert first_new.certificate is None
    assert certified.certificate is not None
    assert certified.certificate.support_frames == 2
    assert certified.certificate.frame_index == 13


def test_ready_episode_is_consumed_only_after_complete_emission_and_commit() -> None:
    tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(2, 0.8))
    _observe(tracker, 1, 0.0)
    update = _observe(tracker, 2, 0.05)
    assert update.certificate is not None
    assert tracker.proposal_allowed

    tracker.record_emission(started=False, completed=False, committed=False)
    assert tracker.proposal_allowed
    tracker.record_emission(started=True, completed=True, committed=True)
    assert tracker.consumed
    assert not tracker.proposal_allowed

    # A lingering READY prompt is the same physical opportunity.
    assert _observe(tracker, 3, 0.1).certificate is None
    _observe(
        tracker,
        4,
        0.15,
        kind=PromptObservationKind.HOOK_INSTRUCTION,
    )
    _observe(tracker, 5, 0.2)
    next_episode = _observe(tracker, 6, 0.25)
    assert next_episode.certificate is not None
    assert next_episode.certificate.physical_ready_episode_id != (
        update.certificate.physical_ready_episode_id
    )


def test_each_physical_ready_certificate_owns_one_bounded_opportunity() -> None:
    tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(1, 0.8))

    for index in range(1000):
        update = _observe(tracker, index * 2 + 1, float(index))
        assert update.certificate is not None
        expected = (
            f"{update.certificate.physical_ready_episode_id}:START_HOOK"
        )
        assert tracker.start_hook_opportunity_id == expected
        assert tracker.proposal_allowed
        tracker.record_proposal()
        assert tracker.proposal_created
        tracker.record_emission(started=True, completed=True, committed=True)
        assert tracker.consumed
        _observe(
            tracker,
            index * 2 + 2,
            float(index) + 0.1,
            kind=PromptObservationKind.WAITING_IN_PROGRESS,
        )

    assert tracker.physical_ready_episode_id is None
    assert tracker.start_hook_opportunity_id is None
    assert not tracker.proposal_created


def test_partial_emission_is_terminal_but_pre_emission_rejection_is_not() -> None:
    tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(1, 0.8))
    assert _observe(tracker, 1, 0.0).certificate is not None
    tracker.record_emission(started=False, completed=False, committed=False)
    assert tracker.proposal_allowed
    tracker.record_emission(started=True, completed=False, committed=False)
    assert not tracker.proposal_allowed
    assert not tracker.consumed


def test_disappeared_ready_certificate_cannot_emit_late() -> None:
    tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(1, 0.8))
    certified = _observe(tracker, 1, 0.0)
    assert certified.certificate is not None
    assert tracker.start_hook_opportunity_id == "ready:1:START_HOOK"

    cleared = _observe(
        tracker,
        2,
        0.1,
        kind=PromptObservationKind.WAITING_IN_PROGRESS,
    )

    assert cleared.certificate is None
    assert tracker.certificate is None
    assert tracker.start_hook_opportunity_id is None
    assert not tracker.proposal_created


def test_single_low_confidence_conflicting_or_panicked_ready_never_certifies() -> None:
    tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(2, 0.8))
    assert _observe(tracker, 1, 0.0, confidence=0.79).certificate is None
    assert _observe(tracker, 2, 0.05, conflict=True).certificate is None
    assert _observe(tracker, 3, 0.1, panic=True).certificate is None
    assert _observe(tracker, 4, 0.15).certificate is None
    assert _observe(tracker, 5, 0.2).certificate is not None


def test_ready_certificate_can_be_evaluated_for_stale_and_downstream_states() -> None:
    for state in (
        RuntimeState.SYNC_REQUIRED,
        RuntimeState.CAST_PENDING,
        RuntimeState.IDLE,
        RuntimeState.HOOK,
        RuntimeState.PRESS,
        RuntimeState.GET,
    ):
        tracker = ReadyRecoveryTracker(ReadyRecoveryConfig(2, 0.8))
        assert _observe(tracker, 1, 0.0, state=state).certificate is None
        assert _observe(tracker, 2, 0.05, state=state).certificate is not None

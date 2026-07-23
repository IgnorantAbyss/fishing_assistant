from __future__ import annotations

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PromptObservation,
    PromptObservationKind,
    ResultBannerObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.live.cast_opportunity import (
    CastOpportunityConfig,
    CastOpportunityController,
    PostCycleClearanceTracker,
)
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.scheduling import MissedReadyRecoveryTracker


def _prompt(
    kind: PromptObservationKind,
    frame: int,
    timestamp: float,
    confidence: float = 0.96,
) -> PromptObservation:
    return PromptObservation(
        kind,
        confidence,
        {kind.value: confidence},
        "session_20260723_150137",
        frame,
        timestamp,
    )


def _bundle(
    frame: int,
    timestamp: float,
    *,
    prompt: PromptObservationKind,
    confidence: float = 0.96,
    hook: bool = False,
    get: bool = False,
    banner: bool | None = None,
) -> ObservationBundle:
    return ObservationBundle(
        frame,
        timestamp,
        _prompt(prompt, frame, timestamp, confidence),
        HookObservation(
            hook,
            0.99 if hook else 0.0,
            frame,
            timestamp,
            fill_ratio=0.4 if hook else None,
            evidence={"matched_features": ["hook_bar_rect", "bar_fill"]}
            if hook else {},
        ),
        None,
        GetObservation(get, 0.99 if get else 0.0, frame, timestamp),
        (
            ResultBannerObservation(
                banner,
                0.99,
                frame,
                timestamp,
            )
            if banner is not None else None
        ),
    )


def _evidence(
    state: RuntimeState | None,
    frame: int,
    timestamp: float,
    confidence: float = 0.96,
) -> StateEvidence:
    return StateEvidence(
        {} if state is None else {state: confidence},
        ("deterministic_session_timeline",),
        (),
        state,
        confidence,
        "deterministic_session_timeline",
        frame,
        timestamp,
    )


def test_waiting_stable_hook_instruction_recovers_without_start_hook() -> None:
    tracker = MissedReadyRecoveryTracker(
        stable_frames=2,
        min_confidence=0.80,
        confirmation_timeout_seconds=0.30,
    )
    fsm = FishingFSM(initial_state=RuntimeState.WAITING)

    first = tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.HOOK_INSTRUCTION, 2491, 114.426),
    )
    assert first.recovered is False
    assert tracker.active is True

    second = tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.HOOK_INSTRUCTION, 2492, 114.476),
    )
    assert second.recovered is True
    recovered = fsm.recover_missed_ready(
        114.476,
        reason="stable_hook_instruction_while_waiting",
    )
    assert recovered.previous_state == RuntimeState.WAITING
    assert recovered.next_state == RuntimeState.HOOK_PENDING
    assert recovered.action_request.intent == ActionIntent.NONE


def test_unknown_idle_low_confidence_and_single_hook_do_not_recover() -> None:
    tracker = MissedReadyRecoveryTracker(
        stable_frames=2,
        min_confidence=0.80,
        confirmation_timeout_seconds=0.30,
    )
    assert tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.UNKNOWN, 1, 1.0),
    ).recovered is False
    assert tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.IDLE_CAST, 2, 1.1),
    ).recovered is False
    assert tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.HOOK_INSTRUCTION, 3, 1.2, 0.79),
    ).recovered is False
    assert tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.HOOK_INSTRUCTION, 4, 1.3),
    ).recovered is False
    assert tracker.observe(
        RuntimeState.WAITING,
        _prompt(PromptObservationKind.UNKNOWN, 5, 1.35),
    ).recovered is False
    assert tracker.active is False


def test_normal_ready_bite_flow_is_unchanged() -> None:
    tracker = MissedReadyRecoveryTracker(
        stable_frames=2,
        min_confidence=0.80,
        confirmation_timeout_seconds=0.30,
    )
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.WAITING,
    )
    prompt = _prompt(PromptObservationKind.READY_BITE, 1, 1.0)
    assert tracker.observe(RuntimeState.WAITING, prompt).recovered is False
    result = fsm.advance(
        _evidence(RuntimeState.READY, 1, 1.0),
        1.0,
        _bundle(1, 1.0, prompt=PromptObservationKind.READY_BITE),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.READY


def test_session_150137_recovers_two_distinct_cycles_and_finishes_five_total() -> None:
    prompt_tracker = MissedReadyRecoveryTracker(
        stable_frames=2,
        min_confidence=0.80,
        confirmation_timeout_seconds=0.30,
    )
    fsm = FishingFSM(
        FSMConfig(
            stable_frames=1,
            result_minimum_pending_sec=0.1,
            result_maximum_pending_sec=2.0,
        ),
        initial_state=RuntimeState.WAITING,
    )
    clearances = PostCycleClearanceTracker(CastOpportunityConfig(
        stable_idle_frames_required=1,
        clearance_freshness_seconds=4.0,
    ))
    opportunities = CastOpportunityController()
    deduplicator = WouldFireDeduplicator()
    recovered_cycle_ids: list[int] = []
    cast_events: list[dict] = []

    # Existing Runtime had already counted the first two cycles. These are the
    # two missing prompt windows copied from session_20260723_150137.
    missed_windows = (
        (114.426, 118.428, 134.482, 2491),
        (170.725, 174.734, 190.791, 3858),
    )
    for index, (hook_at, unknown_at, idle_at, frame) in enumerate(
        missed_windows,
        start=1,
    ):
        if index > 1:
            fsm.force_state(
                RuntimeState.WAITING,
                hook_at - 1.0,
                "manual_cast_waiting_acknowledgement",
            )
        first = prompt_tracker.observe(
            RuntimeState.WAITING,
            _prompt(
                PromptObservationKind.HOOK_INSTRUCTION,
                frame,
                hook_at,
            ),
        )
        assert first.recovered is False
        confirmed_at = hook_at + 0.05
        confirmed = prompt_tracker.observe(
            RuntimeState.WAITING,
            _prompt(
                PromptObservationKind.HOOK_INSTRUCTION,
                frame + 1,
                confirmed_at,
            ),
        )
        assert confirmed.recovered is True
        recovered_cycle_ids.append(deduplicator.begin_recovered_cycle())
        recovered = fsm.recover_missed_ready(
            confirmed_at,
            reason="stable_hook_instruction_while_waiting",
        )
        assert recovered.action_request.intent == ActionIntent.NONE

        hook_transition = fsm.advance(
            _evidence(RuntimeState.HOOK, frame + 2, confirmed_at + 0.05),
            confirmed_at + 0.05,
            _bundle(
                frame + 2,
                confirmed_at + 0.05,
                prompt=PromptObservationKind.HOOK_INSTRUCTION,
                hook=True,
            ),
            recorded_observation=True,
        )
        assert hook_transition.next_state == RuntimeState.HOOK
        unknown = fsm.advance(
            _evidence(None, frame + 3, unknown_at),
            unknown_at,
            _bundle(
                frame + 3,
                unknown_at,
                prompt=PromptObservationKind.UNKNOWN,
            ),
            recorded_observation=True,
        )
        assert unknown.next_state == RuntimeState.HOOK
        pending = fsm.advance(
            _evidence(RuntimeState.IDLE, frame + 4, idle_at),
            idle_at,
            _bundle(
                frame + 4,
                idle_at,
                prompt=PromptObservationKind.IDLE_CAST,
                banner=False,
            ),
            recorded_observation=True,
        )
        assert pending.next_state == RuntimeState.RESULT_PENDING
        clearances.observe(
            timestamp=idle_at,
            previous_state=pending.previous_state,
            current_state=pending.next_state,
            prompt_kind=PromptObservationKind.IDLE_CAST,
            prompt_frame_index=frame + 4,
            get_observation=GetObservation(
                False, 0.99, frame + 4, idle_at
            ),
            get_activation_mode=DetectorActivationMode.BURST,
            result_banner=ResultBannerObservation(
                False, 0.99, frame + 4, idle_at
            ),
            physical_get_episode_open=False,
            runtime_cycle_id=f"cycle:{deduplicator.cycle_id}",
        )
        completed_at = idle_at + 0.2
        completed = fsm.advance(
            _evidence(RuntimeState.IDLE, frame + 5, completed_at),
            completed_at,
            _bundle(
                frame + 5,
                completed_at,
                prompt=PromptObservationKind.IDLE_CAST,
                banner=False,
            ),
            recorded_observation=True,
        )
        assert completed.next_state == RuntimeState.IDLE
        clearances.observe(
            timestamp=completed_at,
            previous_state=completed.previous_state,
            current_state=completed.next_state,
            prompt_kind=PromptObservationKind.IDLE_CAST,
            prompt_frame_index=frame + 5,
            get_observation=GetObservation(
                False, 0.99, frame + 5, completed_at
            ),
            get_activation_mode=DetectorActivationMode.BURST,
            result_banner=ResultBannerObservation(
                False, 0.99, frame + 5, completed_at
            ),
            physical_get_episode_open=False,
            runtime_cycle_id=f"cycle:{deduplicator.cycle_id}",
        )
        clearance = clearances.current(completed_at)
        assert clearance is not None
        attempt, _ = opportunities.schedule(
            timestamp=completed_at,
            clearance_id=clearance.clearance_id,
            runtime_state=RuntimeState.IDLE,
            prompt_kind=PromptObservationKind.IDLE_CAST,
            physical_get_episode_open=False,
        )
        assert attempt is not None
        would_cast = deduplicator.observe(
            ActionRequest(ActionIntent.CAST, 0.99, "certified clearance"),
            safety_reason="action_emission_disabled",
            frame_index=frame + 5,
            timestamp=completed_at,
            runtime_state=RuntimeState.IDLE.value,
            prompt_evidence={"predicted_label": "IDLE_CAST"},
            specialized_evidence={"get": {"detected": False}},
            identity_suffix=attempt.opportunity_id,
        )
        assert would_cast is not None
        cast_events.append(would_cast)
        opportunities.cancel(
            timestamp=completed_at,
            reason="deterministic_no_input_test",
        )
        clearances.consume(clearance.clearance_id)
        deduplicator.finish_cycle()

    assert recovered_cycle_ids == [1, 2]
    assert len(set(recovered_cycle_ids)) == 2
    assert opportunities.summary()["cast_opportunity_count"] == 2
    assert len(cast_events) == 2
    assert deduplicator.unique_events["WOULD_START_HOOK"] == 0
    assert deduplicator.unique_events["WOULD_CAST"] == 2
    missed_ready_recovery_count = 2
    assert 3 + missed_ready_recovery_count == 5

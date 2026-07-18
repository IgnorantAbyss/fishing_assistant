from __future__ import annotations

import pytest

from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.observations import PromptObservation, PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.scheduling import PromptPollingConfig, RuntimeSchedulePolicy


def _prompt(frame: int, timestamp: float, kind: PromptObservationKind) -> PromptObservation:
    confidence = 0.99 if kind != PromptObservationKind.UNKNOWN else 0.0
    return PromptObservation(
        kind, confidence, {kind.value: confidence}, "existing_prompt_observer",
        frame, timestamp,
    )


def _bundle(frame: int, timestamp: float, kind: PromptObservationKind) -> ObservationBundle:
    return ObservationBundle(frame, timestamp, _prompt(frame, timestamp, kind))


def _evidence(
    frame: int, timestamp: float, state: RuntimeState | None,
) -> StateEvidence:
    return StateEvidence(
        {} if state is None else {state: 0.99}, (), (), state,
        0.99 if state else 0.0, "prompt", frame, timestamp,
    )


def _policy() -> RuntimeSchedulePolicy:
    return RuntimeSchedulePolicy(PromptPollingConfig(
        waiting_interval_seconds=4.0,
        waiting_min_seconds=3.0,
        waiting_max_seconds=5.0,
        ready_fps=20.0,
        result_pending_fps=5.0,
        ready_confirmation_timeout_seconds=0.3,
    ))


def test_waiting_keeps_low_frequency_polling_until_ready_candidate() -> None:
    policy = _policy()
    assert policy.prompt_interval_seconds(RuntimeState.WAITING) == 4.0

    update = policy.observe_prompt(
        RuntimeState.WAITING, PromptObservationKind.READY_BITE, 10.0
    )
    assert update.burst_started is True
    assert update.ready_support_frames == 1
    assert policy.prompt_interval_seconds(RuntimeState.WAITING) == 0.05


def test_second_ready_observation_uses_burst_without_waiting_four_seconds() -> None:
    policy = _policy()
    policy.observe_prompt(RuntimeState.WAITING, PromptObservationKind.READY_BITE, 10.0)
    update = policy.observe_prompt(
        RuntimeState.WAITING, PromptObservationKind.READY_BITE, 10.05
    )
    assert update.ready_support_frames == 2
    assert update.candidate_reset_required is False
    assert update.candidate_age_seconds == pytest.approx(0.05)


def test_ready_confirmation_transitions_without_action_then_proposes_on_new_frame() -> None:
    policy = _policy()
    fsm = FishingFSM(
        FSMConfig(stable_frames=2), initial_state=RuntimeState.WAITING,
        initial_timestamp=10.0,
    )

    first_update = policy.observe_prompt(
        fsm.state, PromptObservationKind.READY_BITE, 10.0
    )
    first = fsm.advance(
        _evidence(1, 10.0, RuntimeState.READY), 10.0,
        _bundle(1, 10.0, PromptObservationKind.READY_BITE),
        recorded_observation=True,
    )
    assert first_update.burst_started is True
    assert first.next_state == RuntimeState.WAITING
    assert first.action_request.intent == ActionIntent.NONE

    second_update = policy.observe_prompt(
        fsm.state, PromptObservationKind.READY_BITE, 10.05
    )
    second = fsm.advance(
        _evidence(2, 10.05, RuntimeState.READY), 10.05,
        _bundle(2, 10.05, PromptObservationKind.READY_BITE),
        recorded_observation=True,
    )
    assert second_update.ready_support_frames == 2
    assert second.next_state == RuntimeState.READY
    assert second.action_request.intent == ActionIntent.NONE

    policy.confirm_ready(10.05)
    third = fsm.advance(
        _evidence(3, 10.10, RuntimeState.READY), 10.10,
        _bundle(3, 10.10, PromptObservationKind.READY_BITE),
        recorded_observation=True,
    )
    assert third.next_state == RuntimeState.READY
    assert third.action_request.intent == ActionIntent.START_HOOK


def test_unknown_or_waiting_cancels_single_frame_ready_candidate() -> None:
    for kind in (
        PromptObservationKind.UNKNOWN,
        PromptObservationKind.WAITING_IN_PROGRESS,
    ):
        policy = _policy()
        policy.observe_prompt(RuntimeState.WAITING, PromptObservationKind.READY_BITE, 1.0)
        update = policy.observe_prompt(RuntimeState.WAITING, kind, 1.05)
        assert update.burst_cancelled is True
        assert update.candidate_reset_required is True
        assert policy.prompt_interval_seconds(RuntimeState.WAITING) == 4.0


def test_expired_ready_candidate_restarts_without_reusing_stale_support() -> None:
    policy = _policy()
    policy.observe_prompt(RuntimeState.WAITING, PromptObservationKind.READY_BITE, 2.0)
    update = policy.observe_prompt(
        RuntimeState.WAITING, PromptObservationKind.READY_BITE, 2.31
    )
    assert update.burst_timed_out is True
    assert update.candidate_reset_required is True
    assert update.burst_started is True
    assert update.ready_support_frames == 1


def test_ready_episode_emits_at_most_one_unique_would_start_hook() -> None:
    tracker = WouldFireDeduplicator()
    kwargs = {
        "safety_reason": "action_emission_disabled",
        "timestamp": 1.0,
        "runtime_state": "READY",
        "prompt_evidence": {},
        "specialized_evidence": {},
    }
    from src.fishing_v2.domain.action_intent import ActionRequest

    request = ActionRequest(ActionIntent.START_HOOK, 0.99, "ready_bite_confirmed")
    assert tracker.observe(request, frame_index=1, **kwargs) is not None
    assert tracker.observe(request, frame_index=2, **kwargs) is None
    assert tracker.unique_events == {"WOULD_START_HOOK": 1}

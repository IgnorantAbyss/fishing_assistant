from __future__ import annotations

import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    HookObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.live.hook_action_lifecycle import HookActionLifecycle
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import (
    ActionExecutionMode,
    RuntimeController,
)
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyPolicy


def _qualified_crossing(frame: int, timestamp: float) -> ObservationBundle:
    return ObservationBundle(
        frame,
        timestamp,
        PromptObservation(
            PromptObservationKind.HOOK_INSTRUCTION,
            0.99,
            {PromptObservationKind.HOOK_INSTRUCTION.value: 0.99},
            "session_20260802_062709",
            frame,
            timestamp,
        ),
        HookObservation(
            True,
            0.90,
            frame,
            timestamp,
            fill_ratio=0.75,
            evidence={
                "matched_features": ["hook_bar_rect", "bar_fill"],
                "fallback_ratio_trustworthy": True,
            },
        ),
    )


def _controller(fsm: FishingFSM) -> RuntimeController:
    return RuntimeController(
        ObservationFusion(),
        fsm,
        SafetyPolicy(SafetyConfig(emit_actions=False)),
    )


def _process(controller: RuntimeController, frame: int, timestamp: float):
    return controller.process(
        _qualified_crossing(frame, timestamp),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )


def test_sync_required_hook_recovery_rearms_exactly_once() -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=4729.610886,
        start_hook_applied=True,
    )
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.SYNC_REQUIRED,
        initial_timestamp=4731.714681,
    )

    recovered = fsm.recover_from_sync_required(
        RuntimeState.HOOK,
        4738.242814,
        "qualified_hook_sync_recovery",
    )
    assert recovered.previous_state == RuntimeState.SYNC_REQUIRED
    assert recovered.next_state == RuntimeState.HOOK
    assert lifecycle.rearm_after_sync() is True
    assert fsm.rearm_hook_action_opportunity(4738.242814) is True
    assert lifecycle.rearm_after_sync() is False

    controller = _controller(fsm)
    first = _process(controller, 13512, 4738.268)
    assert first.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert first.safety.reason == "action_emission_disabled"

    controller.discard_external_proposal()
    repeated = _process(controller, 13513, 4738.294)
    assert repeated.fsm.action_request.intent == ActionIntent.NONE


@pytest.mark.parametrize("terminal", ["started", "emitted", "applied"])
def test_sync_recovery_never_rearms_started_or_consumed_action(
    terminal: str,
) -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    if terminal == "started":
        lifecycle.mark_action_started()
    else:
        lifecycle.mark_emission_result(
            emission_started=(terminal == "emitted"),
            applied=(terminal == "applied"),
        )

    assert lifecycle.can_rearm_after_sync() is False
    assert lifecycle.rearm_after_sync() is False
    episode = lifecycle.episode
    assert episode is not None
    assert episode.hook_action_consumed is (terminal != "started")


def test_hook_stall_with_current_qualified_evidence_rearms_once() -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    lifecycle.mark_opportunity_created()

    first = lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        state_age_seconds=3.01,
        qualified_hook_current=True,
        hook_evidence_confidence=0.90,
        hook_evidence_age_seconds=0.01,
    )
    assert first is not None
    assert first.action == "rearm"
    assert lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        state_age_seconds=6.1,
        qualified_hook_current=True,
        hook_evidence_confidence=0.91,
        hook_evidence_age_seconds=0.01,
    ) is None


def test_hook_stall_without_current_evidence_returns_to_sync_required() -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    decision = lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        state_age_seconds=341.06,
        qualified_hook_current=False,
        hook_evidence_confidence=None,
        hook_evidence_age_seconds=None,
    )
    assert decision is not None
    assert decision.action == "sync_required"

    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.HOOK,
    )
    transition = fsm.return_hook_stall_to_sync_required(341.06)
    assert transition.previous_state == RuntimeState.HOOK
    assert transition.next_state == RuntimeState.SYNC_REQUIRED
    assert transition.transition_reason == (
        "hook_action_stall_without_current_hook_evidence"
    )


@pytest.mark.parametrize(
    "blocker",
    (
        "panic_triggered",
        "foreground_not_confirmed",
        "integrity_mismatch",
        "action_not_allowlisted",
    ),
)
def test_hook_stall_safety_blocker_never_rearms_or_consumes(
    blocker: str,
) -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    decision = lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        state_age_seconds=4.0,
        qualified_hook_current=True,
        hook_evidence_confidence=0.90,
        hook_evidence_age_seconds=0.01,
        safety_blockers=(blocker,),
    )
    assert decision is not None
    assert decision.action == "blocked"
    episode = lifecycle.episode
    assert episode is not None
    assert episode.watchdog_triggered is False
    assert episode.hook_action_consumed is False


def test_preserved_sync_cycle_releases_only_unemitted_hook_proposal() -> None:
    tracker = WouldFireDeduplicator()
    request = ActionRequest(ActionIntent.HOOK_ACTION, 0.90, "crossed")
    event = tracker.observe(
        request,
        safety_reason="action_emission_disabled",
        frame_index=1,
        timestamp=1.0,
        runtime_state=RuntimeState.HOOK.value,
        prompt_evidence={},
        specialized_evidence={},
    )
    assert event is not None
    tracker.reset_for_sync_recovery(preserve_cycle=True)
    assert tracker.cycle_id == 1
    tracker.release(ActionIntent.HOOK_ACTION)
    assert tracker.already_consumed(ActionIntent.HOOK_ACTION) is False
    assert tracker.observe(
        request,
        safety_reason="action_emission_disabled",
        frame_index=2,
        timestamp=2.0,
        runtime_state=RuntimeState.HOOK.value,
        prompt_evidence={},
        specialized_evidence={},
    ) is not None


def test_normal_hook_pending_path_is_unchanged() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.HOOK_PENDING,
    )
    result = _process(_controller(fsm), 1, 0.1)
    assert result.fsm.previous_state == RuntimeState.HOOK_PENDING
    assert result.fsm.next_state == RuntimeState.HOOK
    assert result.fsm.action_request.intent == ActionIntent.HOOK_ACTION

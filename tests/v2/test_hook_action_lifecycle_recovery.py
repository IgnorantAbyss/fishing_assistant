from __future__ import annotations

import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    HookObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import (
    ObservationFusion,
    StateEvidence,
)
from src.fishing_v2.live.hook_action_lifecycle import HookActionLifecycle
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import (
    ActionExecutionMode,
    RuntimeController,
)
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyPolicy


def _qualified_crossing(
    frame: int,
    timestamp: float,
    *,
    fill_ratio: float = 0.75,
) -> ObservationBundle:
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
            fill_ratio=fill_ratio,
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


def _commit_start_hook(
    *,
    config: FSMConfig,
    opportunity_id: str = "ready:300:START_HOOK",
) -> tuple[FishingFSM, RuntimeController]:
    fsm = FishingFSM(
        config,
        initial_state=RuntimeState.READY,
        initial_timestamp=-0.1,
    )
    controller = _controller(fsm)
    assert fsm.reconcile_start_hook_opportunity(opportunity_id)
    ready = ObservationBundle(
        1,
        -0.01,
        PromptObservation(
            PromptObservationKind.READY_BITE,
            0.99,
            {PromptObservationKind.READY_BITE.value: 0.99},
            "start-hook-transition-race",
            1,
            -0.01,
        ),
    )
    proposed = controller.process(
        ready,
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert proposed.fsm.action_request.intent == ActionIntent.START_HOOK
    committed = fsm.commit_action(proposed.fsm.action_request, 0.0)
    assert committed.action_applied is True
    assert committed.next_state == RuntimeState.HOOK_PENDING
    return fsm, controller


def _prompt_only(
    frame: int,
    timestamp: float,
    kind: PromptObservationKind,
    *,
    confidence: float = 0.99,
) -> ObservationBundle:
    return ObservationBundle(
        frame,
        timestamp,
        PromptObservation(
            kind,
            confidence,
            {kind.value: confidence},
            "start-hook-transition-race",
            frame,
            timestamp,
        ),
    )


def _process(controller: RuntimeController, frame: int, timestamp: float):
    return controller.process(
        _qualified_crossing(frame, timestamp),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )


@pytest.mark.parametrize("ack_delay_seconds", [0.016, 0.018, 0.021])
def test_live_style_start_hook_conflict_race_waits_for_next_frame_ack(
    ack_delay_seconds: float,
) -> None:
    config = FSMConfig(
        stable_frames=1,
        sync_lost_timeout_sec=2.0,
        hook_pending_timeout_sec=3.0,
    )
    fsm, controller = _commit_start_hook(config=config)

    initial_conflict = controller.process(
        _prompt_only(2, 0.09, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert initial_conflict.fsm.next_state == RuntimeState.HOOK_PENDING

    boundary = controller.process(
        _prompt_only(3, 2.111, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert boundary.fsm.next_state == RuntimeState.HOOK_PENDING
    assert boundary.fsm.transition_reason == (
        "start_hook_transition_conflict_grace"
    )

    acknowledgement_at = 2.111 + ack_delay_seconds
    acknowledged = controller.process(
        _prompt_only(
            4,
            acknowledgement_at,
            PromptObservationKind.HOOK_INSTRUCTION,
        ),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert acknowledged.fsm.next_state == RuntimeState.HOOK_PENDING
    assert acknowledged.fsm.visual_acknowledgement == "HOOK_INSTRUCTION"

    hook_evidence_at = acknowledgement_at + 0.025
    entered = _process(controller, 5, hook_evidence_at)
    assert entered.fsm.previous_state == RuntimeState.HOOK_PENDING
    assert entered.fsm.next_state == RuntimeState.HOOK
    assert entered.fsm.action_request.intent == ActionIntent.HOOK_ACTION

    committed = fsm.commit_action(
        entered.fsm.action_request,
        hook_evidence_at,
    )
    assert committed.action_applied is True
    assert committed.next_state == RuntimeState.RESULT_PENDING
    repeated = _process(controller, 6, hook_evidence_at + 0.026)
    assert repeated.fsm.action_request.intent == ActionIntent.NONE


def test_fresh_qualified_hook_wins_at_start_hook_conflict_boundary() -> None:
    config = FSMConfig(
        stable_frames=1,
        sync_lost_timeout_sec=2.0,
        hook_pending_timeout_sec=3.0,
    )
    fsm, controller = _commit_start_hook(config=config)
    controller.process(
        _prompt_only(2, 0.09, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )

    hook_bundle = _qualified_crossing(3, 2.111)
    conflicting_bundle = ObservationBundle(
        3,
        2.111,
        _prompt_only(
            3,
            2.111,
            PromptObservationKind.READY_BITE,
        ).prompt,
        hook_bundle.hook,
    )
    crossed_fsm = fsm.advance(
        StateEvidence(
            {RuntimeState.HOOK: 0.90},
            ("qualified_hook_bar",),
            ("ready_prompt_overlaps_hook_bar",),
            RuntimeState.HOOK,
            0.90,
            "qualified_hook_bar_with_prompt_overlap",
            3,
            2.111,
        ),
        2.111,
        conflicting_bundle,
        recorded_observation=True,
    )

    assert crossed_fsm.previous_state == RuntimeState.HOOK_PENDING
    assert crossed_fsm.next_state == RuntimeState.HOOK
    assert crossed_fsm.action_request.intent == ActionIntent.HOOK_ACTION


def test_start_hook_transition_grace_remains_bounded() -> None:
    config = FSMConfig(
        stable_frames=1,
        sync_lost_timeout_sec=2.0,
        hook_pending_timeout_sec=3.0,
    )
    fsm, controller = _commit_start_hook(config=config)
    controller.process(
        _prompt_only(2, 0.09, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )

    timed_out = controller.process(
        _prompt_only(3, 3.001, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )

    assert timed_out.fsm.next_state == RuntimeState.SYNC_REQUIRED
    assert timed_out.fsm.transition_reason == "hook_pending_timeout"


def test_non_emitted_hook_pending_conflict_keeps_generic_fail_closed() -> None:
    fsm = FishingFSM(
        FSMConfig(
            stable_frames=1,
            sync_lost_timeout_sec=2.0,
            hook_pending_timeout_sec=3.0,
        ),
        initial_state=RuntimeState.HOOK_PENDING,
        initial_timestamp=0.0,
    )
    controller = _controller(fsm)
    controller.process(
        _prompt_only(1, 0.09, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )

    failed_closed = controller.process(
        _prompt_only(2, 2.111, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )

    assert failed_closed.fsm.next_state == RuntimeState.SYNC_REQUIRED
    assert failed_closed.fsm.transition_reason == (
        "persistent_conflicting_or_illegal_evidence"
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


@pytest.mark.parametrize("terminal", ["emitted", "applied"])
def test_sync_recovery_never_rearms_consumed_action(
    terminal: str,
) -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    lifecycle.mark_emission_result(
        emission_started=(terminal == "emitted"),
        applied=(terminal == "applied"),
    )

    assert lifecycle.can_rearm_after_sync() is False
    assert lifecycle.rearm_after_sync() is False
    episode = lifecycle.episode
    assert episode is not None
    assert episode.hook_action_consumed is True


def test_proposal_without_os_emission_remains_rearmable() -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    lifecycle.mark_action_started()
    lifecycle.mark_emission_result(
        emission_started=False,
        applied=False,
    )

    assert lifecycle.can_rearm_after_sync() is True
    assert lifecycle.episode is not None
    assert lifecycle.episode.hook_action_consumed is False


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
        timestamp=3.01,
        state_age_seconds=3.01,
        qualified_hook_current=True,
        hook_evidence_confidence=0.90,
        hook_evidence_age_seconds=0.01,
    )
    assert first is not None
    assert first.action == "rearm"
    terminal = lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=6.1,
        state_age_seconds=6.1,
        qualified_hook_current=True,
        hook_evidence_confidence=0.91,
        hook_evidence_age_seconds=0.01,
    )
    assert terminal is not None
    assert terminal.action == "sync_required"
    assert terminal.reason == "hook_action_rearm_grace_expired"


def test_hook_stall_without_current_evidence_returns_to_sync_required() -> None:
    lifecycle = HookActionLifecycle(3.0)
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    decision = lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=341.06,
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
        timestamp=4.0,
        state_age_seconds=4.0,
        qualified_hook_current=True,
        hook_evidence_confidence=0.90,
        hook_evidence_age_seconds=0.01,
        safety_blockers=(blocker,),
    )
    assert decision is not None
    assert decision.action == "sync_required"
    episode = lifecycle.episode
    assert episode is not None
    assert episode.terminal_outcome in {
        "panic",
        "foreground_lost",
        "safety_blocked_terminal",
    }
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


def test_hook_instruction_ack_wins_over_pending_prompt_conflict() -> None:
    fsm = FishingFSM(
        FSMConfig(
            stable_frames=1,
            sync_lost_timeout_sec=2.0,
            hook_pending_timeout_sec=3.0,
        ),
        initial_state=RuntimeState.HOOK_PENDING,
        initial_timestamp=0.0,
    )
    controller = _controller(fsm)
    ready = PromptObservation(
        PromptObservationKind.READY_BITE,
        0.99,
        {PromptObservationKind.READY_BITE.value: 0.99},
        "ready-hook-overlap",
        1,
        0.01,
    )
    first = controller.process(
        ObservationBundle(1, 0.01, ready),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert first.fsm.next_state == RuntimeState.HOOK_PENDING

    hook_prompt = PromptObservation(
        PromptObservationKind.HOOK_INSTRUCTION,
        0.99,
        {PromptObservationKind.HOOK_INSTRUCTION.value: 0.99},
        "fresh-hook-ack",
        2,
        2.05,
    )
    acknowledged = controller.process(
        ObservationBundle(2, 2.05, hook_prompt),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert acknowledged.fsm.next_state == RuntimeState.HOOK_PENDING
    assert acknowledged.fsm.transition_reason == (
        "hook_instruction_acknowledged_waiting_for_hook_evidence"
    )
    assert acknowledged.fsm.visual_acknowledgement == "HOOK_INSTRUCTION"

    crossed = _process(controller, 3, 2.10)
    assert crossed.fsm.next_state == RuntimeState.HOOK
    assert crossed.fsm.action_request.intent == ActionIntent.HOOK_ACTION


def test_hook_instruction_ack_does_not_disable_hook_pending_timeout() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, hook_pending_timeout_sec=3.0),
        initial_state=RuntimeState.HOOK_PENDING,
        initial_timestamp=0.0,
    )
    prompt = PromptObservation(
        PromptObservationKind.HOOK_INSTRUCTION,
        0.99,
        {PromptObservationKind.HOOK_INSTRUCTION.value: 0.99},
        "persistent-hook-hint-without-bar",
        1,
        3.01,
    )
    result = _controller(fsm).process(
        ObservationBundle(1, 3.01, prompt),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert result.fsm.next_state == RuntimeState.SYNC_REQUIRED
    assert result.fsm.transition_reason == "hook_pending_timeout"


def test_low_confidence_hook_instruction_is_not_visual_ack() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, prompt_min_confidence=0.8),
        initial_state=RuntimeState.HOOK_PENDING,
    )
    prompt = PromptObservation(
        PromptObservationKind.HOOK_INSTRUCTION,
        0.79,
        {PromptObservationKind.HOOK_INSTRUCTION.value: 0.79},
        "below-existing-prompt-gate",
        1,
        0.1,
    )
    result = _controller(fsm).process(
        ObservationBundle(1, 0.1, prompt),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert result.fsm.visual_acknowledgement is None
    assert result.fsm.transition_reason != (
        "hook_instruction_acknowledged_waiting_for_hook_evidence"
    )


def test_sync_recovery_reuses_same_fresh_qualified_crossing() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.SYNC_REQUIRED,
    )
    controller = _controller(fsm)
    raw = _qualified_crossing(10, 5.0)
    blocked = controller.process(
        raw,
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert blocked.fsm.action_request.intent == ActionIntent.NONE

    fsm.recover_from_sync_required(
        RuntimeState.HOOK,
        5.0,
        "qualified_hook_sync_recovery",
    )
    assert fsm.rearm_hook_action_opportunity(5.0)
    reevaluated = controller.reevaluate_qualified_after_recovery(
        raw,
        blocked.qualified,
        foreground=True,
        runtime_environment_supported=True,
    )
    assert reevaluated.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert reevaluated.safety.reason == "action_emission_disabled"


def test_watchdog_rearm_next_frame_action_ready_exactly_once() -> None:
    lifecycle = HookActionLifecycle(3.0, hard_liveness_ceiling_seconds=12.0)
    lifecycle.begin_episode(
        cycle_id=1,
        timestamp=0.0,
        start_hook_applied=True,
    )
    decision = lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=3.01,
        state_age_seconds=3.01,
        qualified_hook_current=True,
        hook_evidence_confidence=0.90,
        hook_evidence_age_seconds=0.01,
    )
    assert decision is not None and decision.action == "rearm"

    fsm = FishingFSM(
        FSMConfig(stable_frames=1),
        initial_state=RuntimeState.HOOK,
    )
    assert fsm.rearm_hook_action_opportunity(3.01)
    controller = _controller(fsm)
    first = _process(controller, 2, 3.04)
    assert first.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    lifecycle.mark_action_started()
    lifecycle.mark_emission_result(emission_started=True, applied=True)

    controller.discard_external_proposal()
    repeated = _process(controller, 3, 3.07)
    assert repeated.fsm.action_request.intent == ActionIntent.NONE
    assert lifecycle.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=20.0,
        state_age_seconds=20.0,
        qualified_hook_current=True,
        hook_evidence_confidence=0.90,
        hook_evidence_age_seconds=0.01,
    ) is None


def test_rearm_grace_evidence_loss_and_never_ready_are_bounded() -> None:
    lost = HookActionLifecycle(3.0, hard_liveness_ceiling_seconds=12.0)
    lost.begin_episode(cycle_id=1, timestamp=0.0, start_hook_applied=True)
    assert lost.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=3.01,
        state_age_seconds=3.01,
        qualified_hook_current=True,
        hook_evidence_confidence=0.9,
        hook_evidence_age_seconds=0.01,
    ).action == "rearm"
    lost_decision = lost.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=3.04,
        state_age_seconds=3.04,
        qualified_hook_current=False,
        hook_evidence_confidence=None,
        hook_evidence_age_seconds=None,
    )
    assert lost_decision is not None
    assert lost_decision.reason == "hook_action_rearm_grace_evidence_lost"

    never_ready = HookActionLifecycle(
        3.0,
        rearm_grace_seconds=1.0,
        hard_liveness_ceiling_seconds=12.0,
    )
    never_ready.begin_episode(
        cycle_id=2,
        timestamp=0.0,
        start_hook_applied=True,
    )
    assert never_ready.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=3.01,
        state_age_seconds=3.01,
        qualified_hook_current=True,
        hook_evidence_confidence=0.9,
        hook_evidence_age_seconds=0.01,
    ).action == "rearm"
    expired = never_ready.evaluate_stall(
        runtime_state=RuntimeState.HOOK,
        timestamp=4.02,
        state_age_seconds=4.02,
        qualified_hook_current=True,
        hook_evidence_confidence=0.9,
        hook_evidence_age_seconds=0.01,
    )
    assert expired is not None
    assert expired.reason == "hook_action_rearm_grace_expired"


def test_hook_hard_ceiling_and_ten_thousand_ticks_cannot_stay_live() -> None:
    lifecycle = HookActionLifecycle(
        3.0,
        rearm_grace_seconds=3.0,
        hard_liveness_ceiling_seconds=8.0,
    )
    lifecycle.begin_episode(
        cycle_id=53,
        timestamp=0.0,
        start_hook_applied=True,
    )
    terminal = None
    for tick in range(1, 10_001):
        now = tick / 1000.0
        decision = lifecycle.evaluate_stall(
            runtime_state=RuntimeState.HOOK,
            timestamp=now,
            state_age_seconds=now,
            qualified_hook_current=True,
            hook_evidence_confidence=0.9,
            hook_evidence_age_seconds=0.001,
        )
        if decision is not None and decision.action == "sync_required":
            terminal = decision
            break
    assert terminal is not None
    assert terminal.reason in {
        "hook_action_rearm_grace_expired",
        "hook_action_hard_liveness_ceiling",
    }
    assert lifecycle.terminal


def test_new_physical_hook_episode_resets_terminal_watchdog_state() -> None:
    lifecycle = HookActionLifecycle(3.0)
    first = lifecycle.begin_episode(
        cycle_id=7,
        timestamp=0.0,
        start_hook_applied=True,
    )
    lifecycle.mark_sync_required("sync_required")
    second = lifecycle.begin_episode(
        cycle_id=7,
        timestamp=20.0,
        start_hook_applied=True,
    )

    assert second.hook_episode_id != first.hook_episode_id
    assert second.watchdog_phase == "NORMAL"
    assert second.terminal_outcome is None
    assert second.hard_liveness_deadline == 32.0

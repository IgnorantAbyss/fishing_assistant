from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM


def _evidence(state, confidence=0.95, *, conflict=False, frame=1, reason="synthetic"):
    return StateEvidence(
        {} if state is None else {state: confidence}, ("synthetic",),
        ("conflict",) if conflict else (), state, confidence, reason,
        frame, frame * 0.2,
    )


def _bundle(
    frame=1,
    *,
    prompt=PromptObservationKind.UNKNOWN,
    hook=False,
    fill_ratio=None,
    press=False,
    sequence=(),
    get=False,
):
    timestamp = frame * 0.2
    prompt_observation = PromptObservation(
        prompt, 0.95, {prompt.value: 0.95}, "synthetic", frame, timestamp
    )
    return ObservationBundle(
        frame,
        timestamp,
        prompt_observation,
        HookObservation(
            hook, 0.98 if hook else 0.0, frame, timestamp,
            fill_ratio=fill_ratio,
            evidence={"matched_features": ["hook_bar_rect", "bar_fill"] if hook and fill_ratio else ["hook_bar_rect"] if hook else []},
        ),
        PressObservation(press, 0.98 if press else 0.0, frame, timestamp, sequence=tuple(sequence)),
        GetObservation(get, 0.98 if get else 0.0, frame, timestamp),
    )


CONFIG = FSMConfig(
    stable_frames=1,
    cast_pending_timeout_sec=1.0,
    hook_pending_timeout_sec=1.0,
    result_pending_timeout_sec=1.0,
    collect_pending_timeout_sec=1.0,
    sync_lost_timeout_sec=0.5,
    hook_divider_safety_margin_px=10,
    hook_fallback_trigger_threshold=0.70,
    get_retry_interval_seconds=0.4,
    get_max_attempts=12,
    get_max_duration_seconds=5.0,
)


def _cast_request(
    opportunity_id: str | None,
    physical_idle_id: str,
) -> ActionRequest:
    payload = {"physical_idle_id": physical_idle_id}
    if opportunity_id is not None:
        payload["cast_opportunity_id"] = opportunity_id
    return ActionRequest(
        ActionIntent.CAST,
        0.99,
        "armed_cast_lifecycle_ready",
        payload,
    )


def test_idle_cast_pending_waiting_flow_requires_get_guard() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    first = fsm.advance(
        _evidence(RuntimeState.IDLE), 0.1,
        _bundle(prompt=PromptObservationKind.IDLE_CAST),
    )
    assert first.next_state == RuntimeState.IDLE
    assert first.action_request.intent == ActionIntent.CAST
    assert fsm.commit_action(first.action_request, 0.1).next_state == RuntimeState.CAST_PENDING
    second = fsm.advance(
        _evidence(RuntimeState.WAITING, frame=2), 0.2,
        _bundle(2, prompt=PromptObservationKind.WAITING_IN_PROGRESS),
    )
    assert second.next_state == RuntimeState.WAITING


def test_authoritative_idle_new_cast_opportunity_is_not_blocked_by_old_idle_cast() -> None:
    """Regression for production session_20260809_061247 opportunities 159/160."""
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    first = _cast_request("cast_opportunity:159", "physical_idle:159")
    assert fsm.stage_external_cast(first)
    assert fsm.commit_action(first, 1.0).action_applied

    fsm.force_state(RuntimeState.SYNC_REQUIRED, 5.0, "cast_visual_timeout")
    fsm.recover_to_authoritative_idle(
        5.5,
        physical_idle_id="physical_idle:160",
    )
    second = _cast_request("cast_opportunity:160", "physical_idle:160")
    assert fsm.stage_external_cast(second)

    commit = fsm.commit_action(second, 6.1)

    assert commit.action_applied
    assert commit.reason != "action_already_applied_in_state"
    assert commit.next_state == RuntimeState.CAST_PENDING


def test_authoritative_idle_boundary_isolates_legacy_idle_cast_marker() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    legacy = _cast_request(None, "physical_idle:159")
    assert fsm.stage_external_cast(legacy)
    assert fsm.commit_action(legacy, 1.0).action_applied
    assert (RuntimeState.IDLE, ActionIntent.CAST) in fsm.actions_applied

    fsm.force_state(RuntimeState.SYNC_REQUIRED, 5.0, "cast_visual_timeout")
    fsm.recover_to_authoritative_idle(
        5.5,
        physical_idle_id="physical_idle:160",
    )
    fresh = _cast_request("cast_opportunity:160", "physical_idle:160")
    assert fsm.stage_external_cast(fresh)

    assert fsm.commit_action(fresh, 6.1).action_applied
    assert (RuntimeState.IDLE, ActionIntent.CAST) not in fsm.actions_applied


def test_same_cast_opportunity_identity_cannot_commit_twice() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    request = _cast_request("cast_opportunity:160", "physical_idle:160")
    assert fsm.stage_external_cast(request)
    assert fsm.commit_action(request, 1.0).action_applied

    fsm.recover_to_authoritative_idle(
        2.0,
        physical_idle_id="physical_idle:161",
    )
    replayed = _cast_request(
        "cast_opportunity:160",
        "physical_idle:161",
    )
    assert fsm.stage_external_cast(replayed)
    duplicate = fsm.commit_action(replayed, 2.1)

    assert not duplicate.action_applied
    assert duplicate.reason == "cast_opportunity_already_applied"


def test_authoritative_idle_rejects_cast_from_different_physical_episode() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.SYNC_REQUIRED)
    fsm.recover_to_authoritative_idle(
        1.0,
        physical_idle_id="physical_idle:B",
    )

    stale = _cast_request("cast_opportunity:A", "physical_idle:A")

    assert not fsm.stage_external_cast(stale)
    assert fsm.pending_request is None


def test_repeated_same_authoritative_idle_certificate_does_not_reopen_cast() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.SYNC_REQUIRED)
    first_boundary = fsm.recover_to_authoritative_idle(
        1.0,
        physical_idle_id="physical_idle:B",
    )
    request = _cast_request("cast_opportunity:B", "physical_idle:B")
    assert first_boundary.changed
    assert fsm.stage_external_cast(request)
    assert fsm.commit_action(request, 1.1).action_applied

    fsm.force_state(RuntimeState.IDLE, 1.2, "stale_state_pointer_override")
    repeated_boundary = fsm.recover_to_authoritative_idle(
        1.3,
        physical_idle_id="physical_idle:B",
    )
    duplicate = _cast_request("cast_opportunity:B", "physical_idle:B")
    assert not repeated_boundary.changed
    assert repeated_boundary.transition_reason == (
        "authoritative_idle_episode_already_open"
    )
    assert fsm.stage_external_cast(duplicate)
    assert not fsm.commit_action(duplicate, 1.4).action_applied


def test_force_state_without_new_certificate_does_not_clear_legacy_cast_marker() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    first = _cast_request(None, "physical_idle:A")
    assert fsm.stage_external_cast(first)
    assert fsm.commit_action(first, 1.0).action_applied

    fsm.force_state(RuntimeState.SYNC_REQUIRED, 2.0, "test_sync")
    fsm.force_state(RuntimeState.IDLE, 2.1, "state_pointer_only")
    second = _cast_request(None, "physical_idle:B")
    assert fsm.stage_external_cast(second)

    assert not fsm.commit_action(second, 2.2).action_applied


def test_cast_identity_stress_has_no_cross_episode_leakage() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.SYNC_REQUIRED)
    for index in range(1000):
        physical_idle_id = f"physical_idle:{index}"
        opportunity_id = f"cast_opportunity:{index}"
        fsm.recover_to_authoritative_idle(
            float(index),
            physical_idle_id=physical_idle_id,
        )
        request = _cast_request(opportunity_id, physical_idle_id)
        assert fsm.stage_external_cast(request)
        commit = fsm.commit_action(request, float(index) + 0.1)
        assert commit.action_applied
        assert commit.next_state == RuntimeState.CAST_PENDING
        fsm.force_state(
            RuntimeState.SYNC_REQUIRED,
            float(index) + 0.2,
            "cast_visual_timeout",
        )

    assert len(fsm.cast_opportunities_applied) == 1000


def test_get_guard_cancels_cast_and_enters_get() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    result = fsm.advance(
        _evidence(RuntimeState.GET), 0.1,
        _bundle(prompt=PromptObservationKind.IDLE_CAST, get=True),
    )
    assert result.next_state == RuntimeState.GET
    assert result.action_request.intent == ActionIntent.NONE


def test_waiting_ready_hook_pending_flow_arms_before_confirmation() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.WAITING)
    assert fsm.advance(
        _evidence(RuntimeState.READY), 0.1,
        _bundle(prompt=PromptObservationKind.READY_BITE),
    ).next_state == RuntimeState.READY
    intent = fsm.advance(
        _evidence(RuntimeState.READY, frame=2), 0.2,
        _bundle(2, prompt=PromptObservationKind.READY_BITE),
    )
    assert intent.next_state == RuntimeState.READY
    assert intent.action_request.intent == ActionIntent.START_HOOK
    assert fsm.commit_action(intent.action_request, 0.2).next_state == RuntimeState.HOOK_PENDING


def test_hook_instruction_cannot_confirm_hook_without_bar() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.HOOK_PENDING)
    result = fsm.advance(
        _evidence(None), 0.1,
        _bundle(prompt=PromptObservationKind.HOOK_INSTRUCTION),
    )
    assert result.next_state == RuntimeState.HOOK_PENDING


def test_hook_bar_confirms_hook_active() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.HOOK_PENDING)
    result = fsm.advance(_evidence(RuntimeState.HOOK), 0.1, _bundle(hook=True, fill_ratio=0.4))
    assert result.next_state == RuntimeState.HOOK


def test_hook_fallback_threshold_emits_once_without_waiting_for_perfect() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(RuntimeState.HOOK), 0.1,
        _bundle(hook=True, fill_ratio=0.70),
    )
    assert result.action_request.intent == ActionIntent.HOOK_ACTION
    assert result.action_request.payload["fill_ratio"] == 0.70
    assert result.action_request.payload["fallback_used"] is True
    assert result.next_state == RuntimeState.HOOK
    assert 0.70 != 0.95  # The runtime deliberately does not chase the detector's perfect zone.
    assert fsm.commit_action(result.action_request, 0.1).next_state == RuntimeState.RESULT_PENDING
    repeated = fsm.advance(_evidence(None, frame=2), 0.2, _bundle(2, hook=True, fill_ratio=0.70))
    assert repeated.action_request.intent == ActionIntent.NONE


def test_press_panel_confirms_press_and_residual_panel_does_not_repeat() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.RESULT_PENDING)
    entered = fsm.advance(
        _evidence(RuntimeState.PRESS), 0.1,
        _bundle(press=True, sequence=("W", "A"), prompt=PromptObservationKind.PRESS_INSTRUCTION),
    )
    assert entered.next_state == RuntimeState.PRESS
    assert entered.action_request.intent == ActionIntent.PRESS_SEQUENCE
    assert fsm.commit_action(entered.action_request, 0.1).next_state == RuntimeState.RESULT_PENDING
    returned = fsm.advance(
        _evidence(RuntimeState.PRESS, frame=2), 0.2,
        _bundle(2, press=True, sequence=("W", "A")),
    )
    assert returned.next_state == RuntimeState.RESULT_PENDING
    assert returned.action_request.intent == ActionIntent.NONE
    residual = fsm.advance(
        _evidence(RuntimeState.PRESS, frame=3), 0.3,
        _bundle(3, press=True, sequence=("W", "A")),
    )
    assert residual.next_state == RuntimeState.RESULT_PENDING
    assert residual.action_request.intent == ActionIntent.NONE


def test_press_sequence_is_proposed_only_once_even_when_emission_is_disabled() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.PRESS)
    first = fsm.advance(
        _evidence(RuntimeState.PRESS), 0.1,
        _bundle(press=True, sequence=("W", "A", "S", "D")),
    )
    assert first.action_request.intent == ActionIntent.PRESS_SEQUENCE
    second = fsm.advance(
        _evidence(RuntimeState.PRESS, frame=2), 0.2,
        _bundle(2, press=True, sequence=("W", "A", "S", "D")),
    )
    assert second.next_state == RuntimeState.PRESS
    assert second.action_request.intent == ActionIntent.NONE


def test_press_panel_disappearance_without_result_prompt_enters_pending() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.PRESS)
    result = fsm.advance(
        _evidence(None), 0.2,
        _bundle(2, press=False, sequence=()),
    )
    assert result.next_state == RuntimeState.RESULT_PENDING
    assert result.action_request.intent == ActionIntent.NONE


def test_waiting_ignores_idle_prompt_instead_of_leaving_state() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.WAITING)
    result = fsm.advance(
        _evidence(RuntimeState.IDLE), 0.1,
        _bundle(prompt=PromptObservationKind.IDLE_CAST),
    )
    assert result.next_state == RuntimeState.WAITING


def test_cast_pending_idle_prompt_is_visual_grace_until_deadline() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.CAST_PENDING)
    first = fsm.advance(
        _evidence(RuntimeState.IDLE), 0.1,
        _bundle(prompt=PromptObservationKind.IDLE_CAST),
    )
    second = fsm.advance(
        _evidence(RuntimeState.IDLE, frame=2), 0.7,
        _bundle(2, prompt=PromptObservationKind.IDLE_CAST),
    )
    assert first.next_state == RuntimeState.CAST_PENDING
    assert second.next_state == RuntimeState.CAST_PENDING
    assert second.transition_reason == "cast_pending_visual_ack_grace"


def test_live_cast_pending_idle_frame_is_tolerated_for_full_three_point_five_seconds() -> None:
    config = FSMConfig(
        stable_frames=2,
        cast_pending_timeout_sec=4.0,
        sync_lost_timeout_sec=2.0,
    )
    fsm = FishingFSM(config, initial_state=RuntimeState.CAST_PENDING)
    for frame, timestamp in enumerate((0.2, 1.0, 2.2, 3.5), start=1):
        result = fsm.advance(
            _evidence(RuntimeState.IDLE, frame=frame),
            timestamp,
            _bundle(frame, prompt=PromptObservationKind.IDLE_CAST),
        )
        assert result.next_state == RuntimeState.CAST_PENDING
        assert result.transition_reason == "cast_pending_visual_ack_grace"


def test_cast_pending_idle_grace_does_not_hide_high_priority_illegal_evidence() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.CAST_PENDING)
    fsm.advance(
        _evidence(RuntimeState.PRESS, conflict=True), 0.1,
        _bundle(prompt=PromptObservationKind.IDLE_CAST, press=True),
    )
    result = fsm.advance(
        _evidence(RuntimeState.PRESS, conflict=True, frame=2), 0.7,
        _bundle(2, prompt=PromptObservationKind.IDLE_CAST, press=True),
    )
    assert result.next_state == RuntimeState.SYNC_REQUIRED


def test_pending_timeout_requires_sync() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.CAST_PENDING)
    result = fsm.advance(_evidence(None), 1.1, _bundle())
    assert result.next_state == RuntimeState.SYNC_REQUIRED


def test_sync_required_produces_no_action() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.SYNC_REQUIRED)
    result = fsm.advance(_evidence(RuntimeState.IDLE), 0.1, _bundle())
    assert result.next_state == RuntimeState.SYNC_REQUIRED
    assert result.action_request.intent == ActionIntent.NONE


def test_failed_is_telemetry_not_a_runtime_state() -> None:
    assert "FAILURE" not in {state.value for state in RuntimeState}
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.RESULT_PENDING)
    result = fsm.advance(_evidence(None, reason="FAILED detector telemetry"), 0.1, _bundle())
    assert result.next_state == RuntimeState.RESULT_PENDING
    assert result.telemetry == ("FAILED",)


def test_stable_frames_are_required() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=2), initial_state=RuntimeState.WAITING)
    bundle = _bundle(prompt=PromptObservationKind.READY_BITE)
    assert fsm.advance(_evidence(RuntimeState.READY), 0.1, bundle).next_state == RuntimeState.WAITING
    assert fsm.advance(_evidence(RuntimeState.READY, frame=2), 0.2, _bundle(2, prompt=PromptObservationKind.READY_BITE)).next_state == RuntimeState.READY

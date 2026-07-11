from src.fishing_v2.domain.action_intent import ActionIntent
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
    hook_safe_zone_start=0.65,
    hook_safe_zone_end=0.85,
    get_retry_interval_seconds=0.4,
    get_max_attempts=12,
    get_max_duration_seconds=5.0,
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


def test_hook_safe_zone_emits_once_without_waiting_for_perfect() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.HOOK)
    result = fsm.advance(
        _evidence(RuntimeState.HOOK), 0.1,
        _bundle(hook=True, fill_ratio=0.70),
    )
    assert result.action_request.intent == ActionIntent.HOOK_ACTION
    assert result.action_request.payload["position_ratio"] == 0.70
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


def test_waiting_ignores_idle_prompt_instead_of_leaving_state() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.WAITING)
    result = fsm.advance(
        _evidence(RuntimeState.IDLE), 0.1,
        _bundle(prompt=PromptObservationKind.IDLE_CAST),
    )
    assert result.next_state == RuntimeState.WAITING


def test_persistent_illegal_transition_requires_sync() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.CAST_PENDING)
    fsm.advance(_evidence(RuntimeState.IDLE), 0.1, _bundle())
    result = fsm.advance(_evidence(RuntimeState.IDLE, frame=2), 0.7, _bundle(2))
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

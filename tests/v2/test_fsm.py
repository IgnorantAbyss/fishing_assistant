from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM


def _evidence(state, confidence=0.95, *, conflict=False, frame=1):
    return StateEvidence(
        {} if state is None else {state: confidence}, ("synthetic",),
        ("conflict",) if conflict else (), state, confidence, "synthetic",
        frame, frame * 0.2,
    )


CONFIG = FSMConfig(stable_frames=1, cast_pending_timeout_sec=1.0, hook_pending_timeout_sec=1.0, post_catch_timeout_sec=1.0, collect_pending_timeout_sec=1.0, sync_lost_timeout_sec=0.5)


def test_idle_cast_pending_waiting_flow() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    first = fsm.advance(_evidence(RuntimeState.IDLE), 0.1)
    assert first.next_state == RuntimeState.CAST_PENDING
    assert first.action_request.intent == ActionIntent.CAST
    second = fsm.advance(_evidence(RuntimeState.WAITING, frame=2), 0.2)
    assert second.next_state == RuntimeState.WAITING


def test_waiting_ready_hook_pending_hook_flow() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.WAITING)
    assert fsm.advance(_evidence(RuntimeState.READY), 0.1).next_state == RuntimeState.READY
    intent = fsm.advance(_evidence(RuntimeState.READY, frame=2), 0.2)
    assert intent.next_state == RuntimeState.HOOK_PENDING
    assert intent.action_request.intent == ActionIntent.START_HOOK
    assert fsm.advance(_evidence(RuntimeState.HOOK, frame=3), 0.3).next_state == RuntimeState.HOOK


def test_hook_press_post_catch_idle_flow_without_get() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.HOOK)
    press = fsm.advance(_evidence(RuntimeState.PRESS), 0.1)
    assert press.next_state == RuntimeState.PRESS
    assert press.action_request.intent == ActionIntent.PRESS_SEQUENCE
    assert fsm.advance(_evidence(RuntimeState.IDLE, frame=2), 0.2).next_state == RuntimeState.POST_CATCH
    assert fsm.advance(_evidence(RuntimeState.IDLE, frame=3), 0.3).next_state == RuntimeState.IDLE


def test_hook_get_collect_pending_idle_flow() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.HOOK)
    assert fsm.advance(_evidence(RuntimeState.GET), 0.1).next_state == RuntimeState.GET
    collect = fsm.advance(_evidence(RuntimeState.GET, frame=2), 0.2)
    assert collect.next_state == RuntimeState.COLLECT_PENDING
    assert collect.action_request.intent == ActionIntent.COLLECT
    assert fsm.advance(_evidence(RuntimeState.IDLE, frame=3), 0.3).next_state == RuntimeState.IDLE


def test_illegal_transition_is_not_applied() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.WAITING)
    result = fsm.advance(_evidence(RuntimeState.IDLE), 0.1)
    assert result.next_state == RuntimeState.WAITING
    assert result.changed is False


def test_persistent_illegal_transition_requires_sync() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.WAITING)
    fsm.advance(_evidence(RuntimeState.IDLE), 0.1)
    result = fsm.advance(_evidence(RuntimeState.IDLE, frame=2), 0.7)
    assert result.next_state == RuntimeState.SYNC_REQUIRED


def test_pending_timeout_requires_sync() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.CAST_PENDING)
    result = fsm.advance(_evidence(None), 1.1)
    assert result.next_state == RuntimeState.SYNC_REQUIRED


def test_sync_required_produces_no_action() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.SYNC_REQUIRED)
    result = fsm.advance(_evidence(RuntimeState.IDLE), 0.1)
    assert result.next_state == RuntimeState.SYNC_REQUIRED
    assert result.action_request.intent == ActionIntent.NONE


def test_unknown_evidence_does_not_fallback_idle() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.POST_CATCH)
    result = fsm.advance(_evidence(None), 0.1)
    assert result.next_state == RuntimeState.POST_CATCH


def test_action_history_prevents_reproposal_after_force_to_same_state() -> None:
    fsm = FishingFSM(CONFIG, initial_state=RuntimeState.IDLE)
    assert fsm.advance(_evidence(RuntimeState.IDLE), 0.1).action_request.intent == ActionIntent.CAST
    fsm.force_state(RuntimeState.IDLE, 0.2)
    assert fsm.advance(_evidence(RuntimeState.IDLE, frame=2), 0.3).action_request.intent == ActionIntent.NONE


def test_stable_frames_are_required() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=2), initial_state=RuntimeState.WAITING)
    assert fsm.advance(_evidence(RuntimeState.READY), 0.1).next_state == RuntimeState.WAITING
    assert fsm.advance(_evidence(RuntimeState.READY, frame=2), 0.2).next_state == RuntimeState.READY

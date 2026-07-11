import pytest

from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import FusionConfig, ObservationFusion
from src.fishing_v2.perception.observation_bundle import ObservationBundle


def _prompt(kind: PromptObservationKind, confidence: float) -> PromptObservation:
    return PromptObservation(kind, confidence, {kind.value: confidence}, "fake", 1, 0.2, {})


def _bundle(prompt=None, *, hook=0.0, press=0.0, get=0.0):
    return ObservationBundle(
        1, 0.2, prompt,
        HookObservation(hook > 0, hook, 1, 0.2),
        PressObservation(press > 0, press, 1, 0.2),
        GetObservation(get > 0, get, 1, 0.2),
    )


def test_strong_hook_outweighs_residual_ready_prompt() -> None:
    result = ObservationFusion().fuse(
        _bundle(_prompt(PromptObservationKind.READY_BITE, 0.95), hook=0.98),
        RuntimeState.HOOK,
    )
    assert result.recommended_state == RuntimeState.HOOK
    assert result.conflicting_observations == ()
    assert any("compatible_residual_prompt" in item for item in result.supporting_observations)


def test_waiting_does_not_jump_to_idle_on_one_weak_prompt() -> None:
    result = ObservationFusion().fuse(
        _bundle(_prompt(PromptObservationKind.IDLE_CAST, 0.70)),
        RuntimeState.WAITING,
    )
    assert result.recommended_state is None
    assert result.reason == "insufficient_observation_evidence_no_fallback"


def test_press_instruction_and_panel_are_compatible() -> None:
    result = ObservationFusion().fuse(
        _bundle(_prompt(PromptObservationKind.PRESS_INSTRUCTION, 0.95), press=0.98),
        RuntimeState.PRESS,
    )
    assert result.recommended_state == RuntimeState.PRESS
    assert result.conflicting_observations == ()


def test_unknown_prompt_never_falls_back_to_idle() -> None:
    result = ObservationFusion().fuse(
        _bundle(_prompt(PromptObservationKind.UNKNOWN, 0.99)), RuntimeState.RESULT_PENDING
    )
    assert result.recommended_state is None
    assert RuntimeState.IDLE not in result.candidate_states


def test_strong_get_is_independent_candidate() -> None:
    result = ObservationFusion().fuse(_bundle(get=0.93), RuntimeState.HOOK)
    assert result.recommended_state == RuntimeState.GET
    assert result.confidence == pytest.approx(0.93)


def test_conflicting_prompt_is_recorded_not_silently_overwritten() -> None:
    result = ObservationFusion().fuse(
        _bundle(_prompt(PromptObservationKind.IDLE_CAST, 0.95), get=0.96),
        RuntimeState.HOOK,
    )
    assert result.recommended_state == RuntimeState.GET
    assert not result.has_conflict
    assert result.recommended_state == RuntimeState.GET


def test_instruction_prompt_has_no_runtime_state_mapping() -> None:
    result = ObservationFusion().fuse(
        _bundle(_prompt(PromptObservationKind.HOOK_INSTRUCTION, 1.0)), RuntimeState.HOOK_PENDING
    )
    assert result.candidate_states == {}

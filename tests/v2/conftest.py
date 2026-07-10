from pathlib import Path

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


def prompt(kind: PromptObservationKind, confidence: float = 0.95, frame: int = 1) -> PromptObservation:
    return PromptObservation(kind, confidence, {kind.value: confidence}, "scripted", frame, frame * 0.2, {})


def bundle(
    *,
    prompt_observation: PromptObservation | None = None,
    hook: bool = False,
    press: bool = False,
    get: bool = False,
    confidence: float = 0.98,
    frame: int = 1,
) -> ObservationBundle:
    timestamp = frame * 0.2
    return ObservationBundle(
        frame,
        timestamp,
        prompt_observation,
        HookObservation(hook, confidence if hook else 0.0, frame, timestamp),
        PressObservation(press, confidence if press else 0.0, frame, timestamp),
        GetObservation(get, confidence if get else 0.0, frame, timestamp),
    )


def evidence(
    state: RuntimeState | None,
    confidence: float = 0.95,
    *,
    conflict: bool = False,
    frame: int = 1,
) -> StateEvidence:
    return StateEvidence(
        {} if state is None else {state: confidence},
        ("synthetic",),
        ("conflict",) if conflict else (),
        state,
        confidence,
        "synthetic",
        frame,
        frame * 0.2,
    )

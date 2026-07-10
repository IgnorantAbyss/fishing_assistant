from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
)


@dataclass(frozen=True)
class ObservationBundle:
    frame_index: int
    timestamp: float
    prompt: PromptObservation | None = None
    hook: HookObservation | None = None
    press: PressObservation | None = None
    get: GetObservation | None = None

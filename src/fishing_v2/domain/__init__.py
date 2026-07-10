"""Dependency-free v2 domain vocabulary."""

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState

__all__ = [
    "ActionIntent", "ActionRequest", "FrameContext", "GetObservation",
    "HookObservation", "PressObservation", "PromptObservation",
    "PromptObservationKind", "RuntimeState",
]

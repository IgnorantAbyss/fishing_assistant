from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.fishing_v2.domain.action_intent import ActionRequest


@dataclass(frozen=True)
class ActionExecutionContext:
    """Immutable identity and frame context for one proposed action."""

    action_id: str
    episode_id: str
    requested_at: float
    capture_frame_index: int
    runtime_state: str
    target_hwnd: int | None


@dataclass(frozen=True)
class ActionExecutionResult:
    """Result of a complete, rejected, failed, or partial input attempt."""

    action_id: str
    intent_type: str
    requested_at: float
    started_at: float | None
    completed_at: float
    success: bool
    applied: bool
    emitted_event_count: int
    expected_event_count: int
    target_hwnd: int | None
    foreground_hwnd: int | None
    rejection_reason: str | None = None
    error: str | None = None
    partial_execution: bool = False


class ActionSink(Protocol):
    def apply(
        self,
        intent: ActionRequest,
        context: ActionExecutionContext,
    ) -> ActionExecutionResult: ...

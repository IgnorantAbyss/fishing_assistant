from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

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
    """Result of a complete, rejected, failed, or partial input attempt.

    The existing ``applied`` contract means the sink completed every requested
    OS input event. It does not prove that the target application acted on it;
    application acknowledgement remains a later visual/runtime observation.
    """

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
    os_input_emitted: bool = False
    windows_error_code: int | None = None
    windows_error_message: str | None = None
    sendinput_input_count: int = 0
    sendinput_cb_size: int = 0
    input_struct_size: int = 0
    process_architecture: str = "unknown"
    input_mode: str | None = None
    virtual_key: int | None = None
    scan_code: int | None = None
    input_flags: tuple[int, ...] = ()
    integrity_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    attempted_count: int = 0
    completed_key_count: int = 0
    total_key_count: int = 0
    key_timings: tuple[Mapping[str, Any], ...] = ()


class ActionSink(Protocol):
    def apply(
        self,
        intent: ActionRequest,
        context: ActionExecutionContext,
    ) -> ActionExecutionResult: ...

"""One-shot Live CAST scheduling with visual acknowledgement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.ports.action_sink import ActionExecutionResult


@dataclass(frozen=True)
class CastOpportunityConfig:
    visual_ack_timeout_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.visual_ack_timeout_seconds <= 0:
            raise ValueError("cast visual acknowledgement timeout must be positive")


@dataclass(frozen=True)
class CastAttempt:
    opportunity_id: str
    action_id: str
    scheduled_at: float


@dataclass(frozen=True)
class CastOpportunityEvent:
    event_type: str
    payload: Mapping[str, Any]


class CastOpportunityController:
    """Keep one physical CAST opportunity stable across sync/cycle changes."""

    def __init__(self, config: CastOpportunityConfig | None = None) -> None:
        self.config = config or CastOpportunityConfig()
        self._sequence = 0
        self._opportunity_id: str | None = None
        self._open = False
        self._attempted = False
        self._input_completed = False
        self._terminal = False
        self._deadline: float | None = None
        self._opportunity_count = 0
        self._attempt_count = 0
        self._acknowledged_count = 0
        self._timeout_count = 0

    @property
    def opportunity_id(self) -> str | None:
        return self._opportunity_id

    def schedule(
        self,
        *,
        timestamp: float,
        idle_cast_prompt: bool,
        get_observed_absent: bool,
        result_banner_absent: bool,
    ) -> tuple[CastAttempt | None, tuple[CastOpportunityEvent, ...]]:
        if not (idle_cast_prompt and get_observed_absent and result_banner_absent):
            return None, ()
        if self._open:
            return None, ()
        self._sequence += 1
        self._opportunity_id = f"cast_opportunity:{self._sequence}"
        self._open = True
        self._attempted = True
        self._input_completed = False
        self._terminal = False
        self._deadline = None
        self._opportunity_count += 1
        self._attempt_count += 1
        attempt = CastAttempt(
            opportunity_id=self._opportunity_id,
            action_id=f"{self._opportunity_id}:CAST",
            scheduled_at=float(timestamp),
        )
        return attempt, (CastOpportunityEvent("cast_opportunity_started", {
            "opportunity_id": self._opportunity_id,
            "action_id": attempt.action_id,
            "visual_ack_timeout_seconds": self.config.visual_ack_timeout_seconds,
            "os_input_emitted": False,
            "cast_visual_acknowledged": False,
        }),)

    def record_execution(
        self,
        attempt: CastAttempt,
        result: ActionExecutionResult,
        *,
        timestamp: float,
    ) -> tuple[CastOpportunityEvent, ...]:
        if not self._open or attempt.opportunity_id != self._opportunity_id:
            return ()
        complete = bool(
            result.applied
            and result.os_input_emitted
            and result.expected_event_count > 0
            and result.emitted_event_count == result.expected_event_count
            and not result.partial_execution
        )
        self._input_completed = complete
        if complete:
            self._deadline = float(timestamp) + self.config.visual_ack_timeout_seconds
        else:
            # A rejected, zero-event, or partial attempt is terminal. CAST never
            # retries automatically because a partial physical input is ambiguous.
            self._terminal = True
        return (CastOpportunityEvent("cast_attempt_emitted", {
            "opportunity_id": self._opportunity_id,
            "action_id": attempt.action_id,
            "target_hwnd": result.target_hwnd,
            "foreground_hwnd": result.foreground_hwnd,
            "os_input_emitted": result.os_input_emitted,
            "emitted_event_count": result.emitted_event_count,
            "expected_event_count": result.expected_event_count,
            "partial_execution": result.partial_execution,
            "rejection_reason": result.rejection_reason,
            "cast_visual_acknowledged": False,
            "visual_ack_deadline": self._deadline,
        }),)

    def observe(
        self,
        *,
        timestamp: float,
        runtime_state: RuntimeState,
        prompt_kind: PromptObservationKind | None,
    ) -> tuple[CastOpportunityEvent, ...]:
        if not self._open or not self._input_completed:
            return ()
        if (
            runtime_state == RuntimeState.WAITING
            and prompt_kind == PromptObservationKind.WAITING_IN_PROGRESS
        ):
            self._open = False
            self._terminal = True
            self._acknowledged_count += 1
            return (CastOpportunityEvent("cast_visual_acknowledged", {
                "opportunity_id": self._opportunity_id,
                "cast_visual_acknowledged": True,
                "visual_acknowledgement": PromptObservationKind.WAITING_IN_PROGRESS.value,
                "os_input_emitted": False,
            }),)
        if (
            not self._terminal
            and self._deadline is not None
            and float(timestamp) >= self._deadline
        ):
            self._terminal = True
            self._timeout_count += 1
            return (CastOpportunityEvent("cast_visual_timeout", {
                "opportunity_id": self._opportunity_id,
                "cast_visual_acknowledged": False,
                "visual_ack_deadline": self._deadline,
                "retry_scheduled": False,
                "os_input_emitted": False,
            }),)
        return ()

    def summary(self) -> dict[str, Any]:
        return {
            "cast_opportunity_count": self._opportunity_count,
            "cast_attempt_count": self._attempt_count,
            "cast_visual_acknowledged_count": self._acknowledged_count,
            "cast_timeout_count": self._timeout_count,
        }

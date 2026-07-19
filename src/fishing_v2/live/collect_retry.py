"""Visual-acknowledged retry scheduling for the Live COLLECT opportunity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.fishing_v2.ports.action_sink import ActionExecutionResult


@dataclass(frozen=True)
class CollectRetryConfig:
    initial_settle_ms: int = 400
    retry_interval_ms: int = 350
    max_attempts: int = 12
    max_duration_seconds: float = 5.0
    disappearance_confirmation_frames: int = 2

    def __post_init__(self) -> None:
        if self.initial_settle_ms < 0:
            raise ValueError("collect.initial_settle_ms must be non-negative")
        if self.retry_interval_ms < 1:
            raise ValueError("collect.retry_interval_ms must be positive")
        if self.max_attempts < 1:
            raise ValueError("collect.max_attempts must be positive")
        if self.max_duration_seconds <= 0:
            raise ValueError("collect.max_duration_seconds must be positive")
        if self.disappearance_confirmation_frames < 1:
            raise ValueError(
                "collect.disappearance_confirmation_frames must be positive"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CollectRetryConfig":
        return cls(**dict(values or {}))


@dataclass(frozen=True)
class CollectAttempt:
    opportunity_id: str
    attempt_id: str
    attempt_number: int
    scheduled_at: float
    elapsed_seconds: float
    get_confidence: float
    get_confirmation_frames: int


@dataclass(frozen=True)
class CollectRetryEvent:
    event_type: str
    payload: Mapping[str, Any]


class CollectRetryController:
    """Schedule R attempts without equating SendInput completion with game success."""

    def __init__(self, config: CollectRetryConfig | None = None) -> None:
        self.config = config or CollectRetryConfig()
        self._opportunity_id: str | None = None
        self._physical_episode_id: str | None = None
        self._episode_sequence = 0
        self._episode_open = False
        self._runtime_cycle_metadata: str | None = None
        self._appeared_at: float | None = None
        self._next_attempt_at: float | None = None
        self._attempt_count = 0
        self._complete_emission_count = 0
        self._inflight: CollectAttempt | None = None
        self._panel_visible = False
        self._absence_frames = 0
        self._terminal = False
        self._terminal_reason: str | None = None
        self._terminal_counted = False
        self._outcome: str | None = None
        self._last_get_confidence = 0.0
        self._last_get_confirmation_frames = 0
        self._attempt_ids: set[str] = set()
        self._completed_count = 0
        self._visual_timeout_count = 0
        self._visual_acknowledged = False
        self._attempt_counts: dict[str, int] = {}
        self._retry_counts: dict[str, int] = {}
        self._attempt_counts_by_get_episode: dict[str, int] = {}
        self._physical_episode_count = 0
        self._collect_opportunity_count = 0
        self._collect_terminal_episode_count = 0

    @property
    def active(self) -> bool:
        return self._episode_open and not self._terminal

    @property
    def opportunity_id(self) -> str | None:
        return self._opportunity_id

    @property
    def physical_episode_id(self) -> str | None:
        return self._physical_episode_id

    @property
    def visual_acknowledged(self) -> bool:
        return self._visual_acknowledged

    def _elapsed(self, timestamp: float) -> float:
        appeared_at = float(timestamp) if self._appeared_at is None else self._appeared_at
        return max(0.0, float(timestamp) - appeared_at)

    def _base_payload(self, timestamp: float) -> dict[str, Any]:
        return {
            "opportunity_id": self._opportunity_id,
            "physical_get_episode_id": self._physical_episode_id,
            "runtime_cycle_metadata": self._runtime_cycle_metadata,
            "elapsed_seconds": self._elapsed(timestamp),
            "attempt_number": self._attempt_count,
            "get_confidence": self._last_get_confidence,
            "get_confirmation_frames": self._last_get_confirmation_frames,
            "disappearance_confirmation_frames": self._absence_frames,
            "collect_visual_acknowledged": self._visual_acknowledged,
            "next_retry_at": self._next_attempt_at,
        }

    def observe_panel(
        self,
        *,
        opportunity_id: str,
        timestamp: float,
        panel_observed: bool,
        panel_visible: bool,
        get_confidence: float = 0.0,
        get_confirmation_frames: int = 0,
    ) -> tuple[CollectRetryEvent, ...]:
        """Observe a qualified GET result; missing detector runs do not imply absence."""
        events: list[CollectRetryEvent] = []
        if panel_observed and panel_visible and not self._episode_open:
            self._episode_sequence += 1
            self._physical_episode_id = f"get_episode:{self._episode_sequence}"
            self._opportunity_id = f"{self._physical_episode_id}:COLLECT"
            self._runtime_cycle_metadata = opportunity_id
            self._episode_open = True
            self._physical_episode_count += 1
            self._collect_opportunity_count += 1
            self._appeared_at = float(timestamp)
            self._next_attempt_at = float(timestamp) + self.config.initial_settle_ms / 1000.0
            self._attempt_count = 0
            self._complete_emission_count = 0
            self._inflight = None
            self._panel_visible = True
            self._absence_frames = 0
            self._terminal = False
            self._terminal_reason = None
            self._terminal_counted = False
            self._outcome = None
            self._visual_acknowledged = False
            started_payload = {
                **self._base_payload(timestamp),
                "get_confidence": float(get_confidence),
                "get_confirmation_frames": int(get_confirmation_frames),
                "initial_settle_ms": self.config.initial_settle_ms,
            }
            events.extend((
                CollectRetryEvent("get_episode_started", started_payload),
                CollectRetryEvent("collect_retry_started", started_payload),
            ))

        if not self._episode_open or not panel_observed:
            return tuple(events)

        self._last_get_confidence = float(get_confidence)
        self._last_get_confirmation_frames = int(get_confirmation_frames)

        if self._terminal:
            if panel_visible:
                self._panel_visible = True
                self._absence_frames = 0
                return tuple(events)
            self._panel_visible = False
            self._absence_frames += 1
            if self._absence_frames >= self.config.disappearance_confirmation_frames:
                self._episode_open = False
                events.append(CollectRetryEvent("get_episode_completed", {
                    **self._base_payload(timestamp),
                    "terminal_reason": self._terminal_reason,
                    "outcome": self._outcome,
                }))
            return tuple(events)

        if panel_visible:
            self._panel_visible = True
            self._absence_frames = 0
            if self._elapsed(timestamp) >= self.config.max_duration_seconds:
                events.append(self._exhaust(timestamp, "max_duration_exceeded"))
            elif (
                self._attempt_count >= self.config.max_attempts
                and self._next_attempt_at is not None
                and float(timestamp) >= self._next_attempt_at
            ):
                events.append(self._exhaust(timestamp, "max_attempts_exceeded"))
            return tuple(events)

        self._panel_visible = False
        self._absence_frames += 1
        if self._absence_frames < self.config.disappearance_confirmation_frames:
            return tuple(events)
        if self._complete_emission_count > 0:
            self._terminal = True
            self._terminal_reason = "qualified_get_panel_stably_disappeared"
            self._outcome = "visual_acknowledged"
            self._visual_acknowledged = True
            self._completed_count += 1
            self._mark_terminal()
            self._episode_open = False
            events.append(CollectRetryEvent("collect_retry_succeeded", {
                **self._base_payload(timestamp),
                "get_confidence": float(get_confidence),
                "get_confirmation_frames": self._absence_frames,
                "outcome": "visual_acknowledged",
            }))
            events.append(CollectRetryEvent("get_episode_completed", {
                **self._base_payload(timestamp),
                "terminal_reason": self._terminal_reason,
                "outcome": self._outcome,
            }))
        else:
            events.append(self.cancel(timestamp, "panel_disappeared_before_complete_emission"))
            self._episode_open = False
            events.append(CollectRetryEvent("get_episode_completed", {
                **self._base_payload(timestamp),
                "terminal_reason": self._terminal_reason,
                "outcome": self._outcome,
            }))
        return tuple(events)

    def schedule_attempt(
        self,
        *,
        timestamp: float,
        get_confidence: float,
        get_confirmation_frames: int,
    ) -> tuple[CollectAttempt | None, tuple[CollectRetryEvent, ...]]:
        if not self.active or not self._panel_visible or self._inflight is not None:
            return None, ()
        if self._elapsed(timestamp) >= self.config.max_duration_seconds:
            return None, (self._exhaust(timestamp, "max_duration_exceeded"),)
        if self._next_attempt_at is None or float(timestamp) < self._next_attempt_at:
            return None, ()
        if self._attempt_count >= self.config.max_attempts:
            return None, (self._exhaust(timestamp, "max_attempts_exceeded"),)
        attempt_number = self._attempt_count + 1
        attempt_id = f"{self._opportunity_id}:attempt:{attempt_number}"
        if attempt_id in self._attempt_ids:
            return None, ()
        attempt = CollectAttempt(
            opportunity_id=str(self._opportunity_id),
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            scheduled_at=float(timestamp),
            elapsed_seconds=self._elapsed(timestamp),
            get_confidence=float(get_confidence),
            get_confirmation_frames=int(get_confirmation_frames),
        )
        self._attempt_ids.add(attempt_id)
        self._attempt_count = attempt_number
        self._attempt_counts[attempt.opportunity_id] = attempt_number
        self._retry_counts[attempt.opportunity_id] = max(0, attempt_number - 1)
        if self._physical_episode_id is not None:
            self._attempt_counts_by_get_episode[
                self._physical_episode_id
            ] = attempt_number
        self._inflight = attempt
        payload = self.attempt_payload(attempt)
        return attempt, (CollectRetryEvent("collect_attempt_scheduled", payload),)

    def record_execution(
        self,
        attempt: CollectAttempt,
        result: ActionExecutionResult,
        *,
        timestamp: float,
    ) -> tuple[CollectRetryEvent, ...]:
        if self._inflight != attempt:
            return ()
        self._inflight = None
        emitted_payload = {
            **self.attempt_payload(attempt),
            "target_hwnd": result.target_hwnd,
            "foreground_hwnd": result.foreground_hwnd,
            "os_input_emitted": result.os_input_emitted,
            "emitted_event_count": result.emitted_event_count,
            "expected_event_count": result.expected_event_count,
            "partial_execution": result.partial_execution,
            "rejection_reason": result.rejection_reason,
            "windows_error_code": result.windows_error_code,
            "windows_error_message": result.windows_error_message,
            "integrity_diagnostics": dict(result.integrity_diagnostics),
        }
        if (
            result.applied
            and result.os_input_emitted
            and result.emitted_event_count == result.expected_event_count
            and result.expected_event_count > 0
            and not result.partial_execution
        ):
            self._complete_emission_count += 1
            self._next_attempt_at = float(timestamp) + self.config.retry_interval_ms / 1000.0
            emitted_payload["next_retry_at"] = self._next_attempt_at
            return (
                CollectRetryEvent("collect_attempt_emitted", emitted_payload),
                CollectRetryEvent("collect_attempt_waiting_ack", {
                    **emitted_payload,
                    "collect_visual_acknowledged": False,
                }),
            )
        reason = (
            "partial_os_input_not_retried" if result.partial_execution
            else result.rejection_reason or "zero_or_incomplete_os_input_not_retried"
        )
        return (
            CollectRetryEvent("collect_attempt_emitted", emitted_payload),
            self.cancel(timestamp, reason, attempt=attempt),
        )

    def cancel(
        self,
        timestamp: float,
        reason: str,
        *,
        attempt: CollectAttempt | None = None,
    ) -> CollectRetryEvent:
        self._terminal = True
        self._terminal_reason = reason
        self._outcome = "cancelled"
        self._mark_terminal()
        self._inflight = None
        payload = self._base_payload(timestamp)
        if attempt is not None:
            payload.update(self.attempt_payload(attempt))
        payload.update({"cancellation_reason": reason, "outcome": "cancelled"})
        return CollectRetryEvent("collect_retry_cancelled", payload)

    def _exhaust(self, timestamp: float, reason: str) -> CollectRetryEvent:
        self._terminal = True
        self._terminal_reason = reason
        self._outcome = "visual_ack_timeout"
        self._mark_terminal()
        self._inflight = None
        self._visual_timeout_count += 1
        return CollectRetryEvent("collect_retry_exhausted", {
            **self._base_payload(timestamp),
            "cancellation_reason": reason,
            "outcome": "visual_ack_timeout",
        })

    def _mark_terminal(self) -> None:
        if not self._terminal_counted:
            self._collect_terminal_episode_count += 1
            self._terminal_counted = True

    def attempt_payload(self, attempt: CollectAttempt) -> dict[str, Any]:
        return {
            "opportunity_id": attempt.opportunity_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "elapsed_seconds": attempt.elapsed_seconds,
            "get_confidence": attempt.get_confidence,
            "get_confirmation_frames": attempt.get_confirmation_frames,
            "next_retry_at": self._next_attempt_at,
            "collect_visual_acknowledged": self._visual_acknowledged,
            "cancellation_reason": None,
            "os_input_emitted": False,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "collect_visual_acknowledged": self._completed_count > 0,
            "collect_completed_count": self._completed_count,
            "collect_attempt_counts": dict(self._attempt_counts),
            "collect_retry_counts": dict(self._retry_counts),
            "collect_visual_timeout_count": self._visual_timeout_count,
            "collect_outcome": self._outcome,
            "collect_terminal_reason": self._terminal_reason,
            "physical_get_episode_count": self._physical_episode_count,
            "collect_opportunity_count": self._collect_opportunity_count,
            "collect_terminal_episode_count": self._collect_terminal_episode_count,
            "collect_attempt_counts_by_get_episode": dict(
                self._attempt_counts_by_get_episode
            ),
        }

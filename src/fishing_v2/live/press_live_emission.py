"""One-shot PRESS Live emission outcome and visual acknowledgement tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.fishing_v2.ports.action_sink import ActionExecutionResult


@dataclass(frozen=True)
class PressLiveEmissionConfig:
    visual_ack_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.visual_ack_timeout_seconds <= 0:
            raise ValueError(
                "PRESS visual acknowledgement timeout must be positive"
            )


@dataclass(frozen=True)
class PressLiveEvent:
    event_type: str
    payload: Mapping[str, Any]


class PressLiveEmissionTracker:
    """Never retry a physical PRESS episode after its first sink attempt."""

    def __init__(
        self,
        config: PressLiveEmissionConfig | None = None,
    ) -> None:
        self.config = config or PressLiveEmissionConfig()
        self._attempted_episodes: set[int] = set()
        self._awaiting_episode: int | None = None
        self._visual_deadline: float | None = None
        self._terminal_outcomes: dict[int, str] = {}
        self._attempted_count = 0
        self._completed_count = 0
        self._partial_count = 0
        self._failed_count = 0
        self._visual_acknowledged_count = 0
        self._visual_timeout_count = 0

    def begin_attempt(
        self,
        *,
        episode_index: int,
        timestamp: float,
        sequence: tuple[str, ...],
    ) -> tuple[bool, tuple[PressLiveEvent, ...]]:
        if episode_index in self._attempted_episodes:
            return False, (PressLiveEvent(
                "press_emission_blocked",
                {
                    "episode_index": episode_index,
                    "timestamp": timestamp,
                    "sequence": list(sequence),
                    "reason": "press_episode_emission_already_attempted",
                    "action_applied": False,
                },
            ),)
        self._attempted_episodes.add(episode_index)
        self._attempted_count += 1
        return True, (PressLiveEvent(
            "press_emission_attempt_started",
            {
                "episode_index": episode_index,
                "timestamp": timestamp,
                "sequence": list(sequence),
                "total_key_count": len(sequence),
                "action_applied": False,
            },
        ),)

    def record_execution(
        self,
        *,
        episode_index: int,
        timestamp: float,
        execution: ActionExecutionResult,
    ) -> tuple[PressLiveEvent, ...]:
        completed_keys = int(execution.completed_key_count)
        total_keys = int(execution.total_key_count)
        attempted = int(execution.attempted_count)
        base = {
            "episode_index": episode_index,
            "timestamp": timestamp,
            "action_id": execution.action_id,
            "attempted_count": attempted,
            "completed_key_count": completed_keys,
            "total_key_count": total_keys,
            "partial_execution": execution.partial_execution,
            "error": execution.error,
            "rejection_reason": execution.rejection_reason,
            "action_applied": execution.applied,
        }
        if execution.applied:
            self._completed_count += 1
            self._awaiting_episode = episode_index
            self._visual_deadline = (
                float(timestamp)
                + self.config.visual_ack_timeout_seconds
            )
            self._terminal_outcomes[episode_index] = (
                "completed_waiting_visual_ack"
            )
            return (PressLiveEvent("press_emission_completed", {
                **base,
                "terminal_outcome": "completed_waiting_visual_ack",
                "visual_ack_deadline": self._visual_deadline,
            }),)
        if execution.partial_execution:
            self._partial_count += 1
            outcome = "partial_not_retried"
            event_type = "press_emission_partial"
        else:
            self._failed_count += 1
            outcome = "failed_not_retried"
            event_type = "press_emission_failed"
        self._terminal_outcomes[episode_index] = outcome
        return (PressLiveEvent(event_type, {
            **base,
            "terminal_outcome": outcome,
        }),)

    def observe_panel(
        self,
        *,
        timestamp: float,
        panel_observed: bool,
        panel_disappeared: bool,
    ) -> tuple[PressLiveEvent, ...]:
        if self._awaiting_episode is None:
            return ()
        episode = self._awaiting_episode
        if panel_observed and panel_disappeared:
            self._visual_acknowledged_count += 1
            self._terminal_outcomes[episode] = "visual_acknowledged"
            self._awaiting_episode = None
            deadline = self._visual_deadline
            self._visual_deadline = None
            return (PressLiveEvent("press_visual_acknowledged", {
                "episode_index": episode,
                "timestamp": timestamp,
                "visual_ack_deadline": deadline,
                "terminal_outcome": "visual_acknowledged",
                "action_applied": False,
            }),)
        if (
            self._visual_deadline is not None
            and float(timestamp) >= self._visual_deadline
        ):
            self._visual_timeout_count += 1
            self._terminal_outcomes[episode] = (
                "visual_ack_timeout_not_retried"
            )
            deadline = self._visual_deadline
            self._awaiting_episode = None
            self._visual_deadline = None
            return (PressLiveEvent("press_visual_ack_timeout", {
                "episode_index": episode,
                "timestamp": timestamp,
                "visual_ack_deadline": deadline,
                "terminal_outcome": "visual_ack_timeout_not_retried",
                "action_applied": False,
            }),)
        return ()

    def summary(self) -> dict[str, Any]:
        return {
            "press_live_emission_attempted_count": self._attempted_count,
            "press_live_emission_completed_count": self._completed_count,
            "press_live_emission_partial_count": self._partial_count,
            "press_live_emission_failed_count": self._failed_count,
            "press_live_visual_acknowledged_count": (
                self._visual_acknowledged_count
            ),
            "press_live_visual_timeout_count": self._visual_timeout_count,
            "press_live_awaiting_visual_ack": (
                self._awaiting_episode is not None
            ),
            "press_live_visual_ack_deadline": self._visual_deadline,
            "press_live_terminal_outcomes": {
                str(key): value
                for key, value in sorted(self._terminal_outcomes.items())
            },
        }

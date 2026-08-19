"""One-shot PRESS Live emission outcome and visual acknowledgement tracking."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Protocol

from src.fishing_v2.ports.action_sink import ActionExecutionResult
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence


@dataclass(frozen=True)
class PressLiveEmissionConfig:
    visual_ack_timeout_seconds: float = 3.0
    initial_delay_min_ms: int = 300
    initial_delay_max_ms: int = 500
    inter_key_gap_min_ms: int = 90
    inter_key_gap_max_ms: int = 170
    key_hold_ms: int = 40

    def __post_init__(self) -> None:
        if self.visual_ack_timeout_seconds <= 0:
            raise ValueError(
                "PRESS visual acknowledgement timeout must be positive"
            )
        for name in (
            "initial_delay_min_ms",
            "initial_delay_max_ms",
            "inter_key_gap_min_ms",
            "inter_key_gap_max_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.initial_delay_min_ms > self.initial_delay_max_ms:
            raise ValueError("PRESS initial delay range is inverted")
        if self.inter_key_gap_min_ms > self.inter_key_gap_max_ms:
            raise ValueError("PRESS inter-key gap range is inverted")
        if self.key_hold_ms < 1:
            raise ValueError("PRESS key hold must be positive")


class PressTimingRng(Protocol):
    def randint(self, lower: int, upper: int) -> int: ...


@dataclass(frozen=True)
class PressTimingPlan:
    sampled_initial_delay_ms: int
    key_hold_ms: tuple[int, ...]
    inter_key_gap_ms: tuple[int, ...]
    planned_total_duration_ms: int

    def payload(self) -> dict[str, Any]:
        return {
            "sampled_initial_delay_ms": self.sampled_initial_delay_ms,
            "key_hold_ms": list(self.key_hold_ms),
            "inter_key_gap_ms": list(self.inter_key_gap_ms),
            "planned_total_duration_ms": self.planned_total_duration_ms,
        }

    def console_schedule(self, sequence: tuple[str, ...]) -> str:
        """Render the complete immutable plan sampled before emission."""
        holds = ",".join(str(value) for value in self.key_hold_ms)
        gaps = ",".join(str(value) for value in self.inter_key_gap_ms)
        return (
            "PRESS scheduled: "
            f"sequence={''.join(sequence)} "
            f"initial_delay_ms={self.sampled_initial_delay_ms} "
            f"hold_ms=[{holds}] "
            f"gap_ms=[{gaps}] "
            f"planned_total_duration_ms={self.planned_total_duration_ms}"
        )


@dataclass(frozen=True)
class ScheduledPressEmission:
    episode_index: int
    sequence: tuple[str, ...]
    slot_capacity: int
    scheduled_at: float
    deadline: float
    timing: PressTimingPlan
    frozen_evidence: StateEvidence | None = None


def pending_press_cancellation_reason(
    pending: ScheduledPressEmission,
    *,
    runtime_state: RuntimeState,
    active_episode: bool,
    episode_index: int,
    panel_disappeared: bool,
    foreground: bool | None,
    panic_triggered: bool,
) -> str | None:
    """Fail closed when a scheduled PRESS opportunity loses eligibility."""
    if runtime_state != RuntimeState.PRESS:
        return "runtime_left_press"
    if panel_disappeared:
        return "press_panel_disappeared"
    if not active_episode or episode_index != pending.episode_index:
        return "press_episode_changed"
    if foreground is not True:
        return "foreground_not_confirmed"
    if panic_triggered:
        return "panic_triggered"
    return None


@dataclass(frozen=True)
class PressLiveEvent:
    event_type: str
    payload: Mapping[str, Any]


class PressLiveEmissionTracker:
    """Never retry a physical PRESS episode after its first sink attempt."""

    def __init__(
        self,
        config: PressLiveEmissionConfig | None = None,
        *,
        rng: PressTimingRng | None = None,
    ) -> None:
        self.config = config or PressLiveEmissionConfig()
        self.rng = rng or random.Random()
        self._attempted_episodes: set[int] = set()
        self._reserved_episodes: set[int] = set()
        self._pending: ScheduledPressEmission | None = None
        self._awaiting_episode: int | None = None
        self._visual_deadline: float | None = None
        self._terminal_outcomes: dict[int, str] = {}
        self._attempted_count = 0
        self._completed_count = 0
        self._partial_count = 0
        self._failed_count = 0
        self._visual_acknowledged_count = 0
        self._visual_timeout_count = 0
        self._scheduled_count = 0
        self._cancelled_count = 0

    @property
    def pending(self) -> ScheduledPressEmission | None:
        return self._pending

    def authoritative_episode_progress(
        self, episode_index: int
    ) -> dict[str, bool]:
        """Expose existing one-shot ownership for anomaly classification."""
        episode = int(episode_index)
        terminal = self._terminal_outcomes.get(episode, "")
        return {
            "opportunity_scheduled": episode in self._reserved_episodes,
            "emission_started": episode in self._attempted_episodes,
            "action_applied": terminal in {
                "completed_waiting_visual_ack",
                "visual_acknowledged",
                "visual_ack_timeout_not_retried",
            },
        }

    def _timing_plan(self, sequence: tuple[str, ...]) -> PressTimingPlan:
        initial = self.rng.randint(
            self.config.initial_delay_min_ms,
            self.config.initial_delay_max_ms,
        )
        holds = tuple(self.config.key_hold_ms for _ in sequence)
        gaps = tuple(
            self.rng.randint(
                self.config.inter_key_gap_min_ms,
                self.config.inter_key_gap_max_ms,
            )
            for _ in range(max(0, len(sequence) - 1))
        )
        return PressTimingPlan(
            initial,
            holds,
            gaps,
            initial + sum(holds) + sum(gaps),
        )

    def schedule(
        self,
        *,
        episode_index: int,
        timestamp: float,
        sequence: tuple[str, ...],
        slot_capacity: int,
        frozen_evidence: StateEvidence | None = None,
    ) -> tuple[ScheduledPressEmission | None, tuple[PressLiveEvent, ...]]:
        if self._pending is not None:
            return None, (PressLiveEvent(
                "press_schedule_blocked",
                {
                    "episode_index": episode_index,
                    "timestamp": timestamp,
                    "reason": "another_press_schedule_is_pending",
                    "action_applied": False,
                },
            ),)
        if episode_index in self._reserved_episodes:
            return None, (PressLiveEvent(
                "press_schedule_blocked",
                {
                    "episode_index": episode_index,
                    "timestamp": timestamp,
                    "reason": "press_episode_opportunity_already_reserved",
                    "action_applied": False,
                },
            ),)
        frozen_sequence = tuple(sequence)
        frozen_slot_capacity = int(slot_capacity)
        timing = self._timing_plan(frozen_sequence)
        pending = ScheduledPressEmission(
            episode_index,
            frozen_sequence,
            frozen_slot_capacity,
            float(timestamp),
            float(timestamp) + timing.sampled_initial_delay_ms / 1000.0,
            timing,
            frozen_evidence,
        )
        self._reserved_episodes.add(episode_index)
        self._pending = pending
        self._scheduled_count += 1
        return pending, (PressLiveEvent(
            "press_emission_scheduled",
            {
                "episode_index": episode_index,
                "timestamp": timestamp,
                "deadline": pending.deadline,
                "sequence": list(frozen_sequence),
                "slot_capacity": frozen_slot_capacity,
                **timing.payload(),
                "action_applied": False,
            },
        ),)

    def due(self, timestamp: float) -> bool:
        return bool(
            self._pending is not None
            and float(timestamp) + 1e-9 >= self._pending.deadline
        )

    def cancel_pending(
        self,
        *,
        timestamp: float,
        reason: str,
    ) -> tuple[PressLiveEvent, ...]:
        pending = self._pending
        if pending is None:
            return ()
        self._pending = None
        self._cancelled_count += 1
        self._terminal_outcomes[pending.episode_index] = (
            f"cancelled_not_retried:{reason}"
        )
        return (PressLiveEvent("press_emission_cancelled", {
            "episode_index": pending.episode_index,
            "timestamp": timestamp,
            "sequence": list(pending.sequence),
            "reason": reason,
            **pending.timing.payload(),
            "terminal_outcome": "cancelled_not_retried",
            "action_applied": False,
        }),)

    def begin_scheduled_attempt(
        self,
        *,
        timestamp: float,
    ) -> tuple[ScheduledPressEmission | None, tuple[PressLiveEvent, ...]]:
        pending = self._pending
        if pending is None or not self.due(timestamp):
            return None, ()
        self._pending = None
        started, events = self.begin_attempt(
            episode_index=pending.episode_index,
            timestamp=timestamp,
            sequence=pending.sequence,
        )
        return (pending if started else None), events

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
        self._reserved_episodes.add(episode_index)
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
            "press_live_emission_scheduled_count": self._scheduled_count,
            "press_live_emission_cancelled_count": self._cancelled_count,
            "press_live_emission_pending": self._pending is not None,
            "press_live_emission_deadline": (
                self._pending.deadline if self._pending is not None else None
            ),
            "press_live_awaiting_visual_ack": (
                self._awaiting_episode is not None
            ),
            "press_live_visual_ack_deadline": self._visual_deadline,
            "press_live_terminal_outcomes": {
                str(key): value
                for key, value in sorted(self._terminal_outcomes.items())
            },
        }

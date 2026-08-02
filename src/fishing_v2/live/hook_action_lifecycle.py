"""Explicit Live HOOK_ACTION lifecycle across synchronization recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.fishing_v2.domain.runtime_state import RuntimeState


@dataclass
class HookActionEpisode:
    cycle_id: int
    hook_episode_id: str
    started_at: float
    start_hook_applied: bool
    hook_action_opportunity_created: bool = False
    hook_action_started: bool = False
    hook_action_emission_started: bool = False
    hook_action_applied: bool = False
    hook_action_consumed: bool = False
    sync_rearmed: bool = False
    watchdog_triggered: bool = False
    watchdog_rearmed: bool = False


@dataclass(frozen=True)
class HookStallDecision:
    action: str
    reason: str
    state_age_seconds: float
    hook_evidence_confidence: float | None
    hook_evidence_age_seconds: float | None
    safety_blockers: tuple[str, ...] = ()


class HookActionLifecycle:
    """Track one physical Hook episode without treating proposal as emission."""

    def __init__(self, stall_timeout_seconds: float = 3.0) -> None:
        if stall_timeout_seconds <= 0:
            raise ValueError("Hook action stall timeout must be positive")
        self.stall_timeout_seconds = float(stall_timeout_seconds)
        self._serial = 0
        self._episode: HookActionEpisode | None = None

    @property
    def episode(self) -> HookActionEpisode | None:
        return self._episode

    def begin_episode(
        self,
        *,
        cycle_id: int,
        timestamp: float,
        start_hook_applied: bool,
    ) -> HookActionEpisode:
        current = self._episode
        if current is not None and current.cycle_id == int(cycle_id):
            current.start_hook_applied = bool(
                current.start_hook_applied or start_hook_applied
            )
            return current
        self._serial += 1
        self._episode = HookActionEpisode(
            int(cycle_id),
            f"cycle:{int(cycle_id)}:hook:{self._serial}",
            float(timestamp),
            bool(start_hook_applied),
        )
        return self._episode

    def finish_episode(self) -> None:
        self._episode = None

    def mark_opportunity_created(self) -> None:
        if self._episode is not None:
            self._episode.hook_action_opportunity_created = True

    def mark_action_started(self) -> None:
        if self._episode is not None:
            self._episode.hook_action_started = True

    def mark_emission_result(
        self,
        *,
        emission_started: bool,
        applied: bool,
    ) -> None:
        if self._episode is None:
            return
        self._episode.hook_action_emission_started = bool(
            self._episode.hook_action_emission_started or emission_started
        )
        self._episode.hook_action_applied = bool(
            self._episode.hook_action_applied or applied
        )
        self._episode.hook_action_consumed = bool(
            self._episode.hook_action_emission_started
            or self._episode.hook_action_applied
        )

    def can_rearm_after_sync(self) -> bool:
        episode = self._episode
        return bool(
            episode is not None
            and episode.start_hook_applied
            and not episode.hook_action_started
            and not episode.hook_action_emission_started
            and not episode.hook_action_applied
            and not episode.hook_action_consumed
        )

    def rearm_after_sync(self) -> bool:
        if not self.can_rearm_after_sync():
            return False
        assert self._episode is not None
        if self._episode.sync_rearmed:
            return False
        self._episode.sync_rearmed = True
        self._episode.hook_action_opportunity_created = True
        return True

    def evaluate_stall(
        self,
        *,
        runtime_state: RuntimeState,
        state_age_seconds: float,
        qualified_hook_current: bool,
        hook_evidence_confidence: float | None,
        hook_evidence_age_seconds: float | None,
        safety_blockers: tuple[str, ...] = (),
    ) -> HookStallDecision | None:
        episode = self._episode
        if (
            runtime_state != RuntimeState.HOOK
            or episode is None
            or episode.hook_action_started
            or episode.hook_action_emission_started
            or episode.hook_action_applied
            or episode.hook_action_consumed
            or float(state_age_seconds) < self.stall_timeout_seconds
        ):
            return None
        if safety_blockers:
            return HookStallDecision(
                "blocked",
                "hook_action_stall_recovery_safety_blocked",
                float(state_age_seconds),
                hook_evidence_confidence,
                hook_evidence_age_seconds,
                tuple(safety_blockers),
            )
        if episode.watchdog_triggered:
            return None
        episode.watchdog_triggered = True
        if qualified_hook_current:
            episode.watchdog_rearmed = True
            episode.hook_action_opportunity_created = True
            return HookStallDecision(
                "rearm",
                "qualified_current_hook_evidence",
                float(state_age_seconds),
                hook_evidence_confidence,
                hook_evidence_age_seconds,
            )
        return HookStallDecision(
            "sync_required",
            "hook_action_stall_without_current_hook_evidence",
            float(state_age_seconds),
            hook_evidence_confidence,
            hook_evidence_age_seconds,
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self._episode) if self._episode is not None else {
            "cycle_id": None,
            "hook_episode_id": None,
            "start_hook_applied": False,
            "hook_action_opportunity_created": False,
            "hook_action_started": False,
            "hook_action_emission_started": False,
            "hook_action_applied": False,
            "hook_action_consumed": False,
            "sync_rearmed": False,
            "watchdog_triggered": False,
            "watchdog_rearmed": False,
        }

"""Explicit Live HOOK_ACTION lifecycle across synchronization recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from src.fishing_v2.domain.runtime_state import RuntimeState


class HookWatchdogPhase(str, Enum):
    NORMAL = "NORMAL"
    REARM_GRACE = "REARM_GRACE"
    TERMINAL = "TERMINAL"


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
    watchdog_phase: str = HookWatchdogPhase.NORMAL.value
    watchdog_rearmed_at: float | None = None
    watchdog_deadline: float | None = None
    hard_liveness_deadline: float | None = None
    terminal_outcome: str | None = None


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

    def __init__(
        self,
        stall_timeout_seconds: float = 3.0,
        *,
        rearm_grace_seconds: float | None = None,
        hard_liveness_ceiling_seconds: float = 12.0,
    ) -> None:
        if stall_timeout_seconds <= 0:
            raise ValueError("Hook action stall timeout must be positive")
        self.stall_timeout_seconds = float(stall_timeout_seconds)
        self.rearm_grace_seconds = float(
            rearm_grace_seconds
            if rearm_grace_seconds is not None
            else stall_timeout_seconds
        )
        self.hard_liveness_ceiling_seconds = float(
            hard_liveness_ceiling_seconds
        )
        if self.rearm_grace_seconds <= 0:
            raise ValueError("Hook rearm grace must be positive")
        if self.hard_liveness_ceiling_seconds <= 0:
            raise ValueError("Hook hard liveness ceiling must be positive")
        self._serial = 0
        self._episode: HookActionEpisode | None = None

    @property
    def episode(self) -> HookActionEpisode | None:
        return self._episode

    @property
    def terminal(self) -> bool:
        return bool(
            self._episode is not None
            and self._episode.terminal_outcome is not None
        )

    def begin_episode(
        self,
        *,
        cycle_id: int,
        timestamp: float,
        start_hook_applied: bool,
    ) -> HookActionEpisode:
        current = self._episode
        new_physical_episode = bool(
            current is not None
            and current.cycle_id == int(cycle_id)
            and current.terminal_outcome is not None
            and start_hook_applied
        )
        if (
            current is not None
            and current.cycle_id == int(cycle_id)
            and not new_physical_episode
        ):
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
            hard_liveness_deadline=(
                float(timestamp) + self.hard_liveness_ceiling_seconds
            ),
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
        if self._episode.hook_action_consumed:
            self._episode.watchdog_phase = (
                HookWatchdogPhase.TERMINAL.value
            )
            self._episode.terminal_outcome = (
                "action_applied" if applied else "emission_started"
            )
        elif not emission_started:
            # Sink-side rejection before any OS input is not consumption.
            self._episode.hook_action_started = False

    def can_rearm_after_sync(
        self,
        *,
        authoritative_hook_evidence: bool = False,
    ) -> bool:
        episode = self._episode
        return bool(
            episode is not None
            and (
                episode.start_hook_applied
                or authoritative_hook_evidence
            )
            and not episode.hook_action_emission_started
            and not episode.hook_action_applied
            and not episode.hook_action_consumed
            and episode.terminal_outcome is None
        )

    def rearm_after_sync(
        self,
        *,
        authoritative_hook_evidence: bool = False,
    ) -> bool:
        if not self.can_rearm_after_sync(
            authoritative_hook_evidence=authoritative_hook_evidence
        ):
            return False
        assert self._episode is not None
        if self._episode.sync_rearmed:
            return False
        self._episode.sync_rearmed = True
        self._episode.hook_action_opportunity_created = True
        return True

    def mark_sync_required(self, outcome: str) -> None:
        if self._episode is None:
            return
        self._episode.watchdog_phase = HookWatchdogPhase.TERMINAL.value
        self._episode.terminal_outcome = str(outcome)

    def _sync_decision(
        self,
        *,
        reason: str,
        state_age_seconds: float,
        hook_evidence_confidence: float | None,
        hook_evidence_age_seconds: float | None,
        safety_blockers: tuple[str, ...] = (),
        terminal_outcome: str = "sync_required",
    ) -> HookStallDecision:
        assert self._episode is not None
        self._episode.watchdog_phase = HookWatchdogPhase.TERMINAL.value
        self._episode.terminal_outcome = terminal_outcome
        return HookStallDecision(
            "sync_required",
            reason,
            float(state_age_seconds),
            hook_evidence_confidence,
            hook_evidence_age_seconds,
            tuple(safety_blockers),
        )

    def evaluate_stall(
        self,
        *,
        runtime_state: RuntimeState,
        timestamp: float,
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
            or episode.terminal_outcome is not None
        ):
            return None
        now = float(timestamp)
        if (
            episode.hard_liveness_deadline is not None
            and now >= episode.hard_liveness_deadline
        ):
            return self._sync_decision(
                reason="hook_action_hard_liveness_ceiling",
                state_age_seconds=state_age_seconds,
                hook_evidence_confidence=hook_evidence_confidence,
                hook_evidence_age_seconds=hook_evidence_age_seconds,
                terminal_outcome="sync_required",
            )
        if episode.watchdog_phase == HookWatchdogPhase.REARM_GRACE.value:
            if safety_blockers:
                outcome = (
                    "panic" if "panic_triggered" in safety_blockers
                    else "foreground_lost"
                    if "foreground_not_confirmed" in safety_blockers
                    else "safety_blocked_terminal"
                )
                return self._sync_decision(
                    reason="hook_action_rearm_grace_safety_blocked",
                    state_age_seconds=state_age_seconds,
                    hook_evidence_confidence=hook_evidence_confidence,
                    hook_evidence_age_seconds=hook_evidence_age_seconds,
                    safety_blockers=safety_blockers,
                    terminal_outcome=outcome,
                )
            if not qualified_hook_current:
                return self._sync_decision(
                    reason="hook_action_rearm_grace_evidence_lost",
                    state_age_seconds=state_age_seconds,
                    hook_evidence_confidence=hook_evidence_confidence,
                    hook_evidence_age_seconds=hook_evidence_age_seconds,
                )
            if (
                episode.watchdog_deadline is not None
                and now >= episode.watchdog_deadline
            ):
                return self._sync_decision(
                    reason="hook_action_rearm_grace_expired",
                    state_age_seconds=state_age_seconds,
                    hook_evidence_confidence=hook_evidence_confidence,
                    hook_evidence_age_seconds=hook_evidence_age_seconds,
                )
            return None
        if float(state_age_seconds) < self.stall_timeout_seconds:
            return None
        if safety_blockers:
            outcome = (
                "panic" if "panic_triggered" in safety_blockers
                else "foreground_lost"
                if "foreground_not_confirmed" in safety_blockers
                else "safety_blocked_terminal"
            )
            return self._sync_decision(
                reason="hook_action_stall_recovery_safety_blocked",
                state_age_seconds=state_age_seconds,
                hook_evidence_confidence=hook_evidence_confidence,
                hook_evidence_age_seconds=hook_evidence_age_seconds,
                safety_blockers=safety_blockers,
                terminal_outcome=outcome,
            )
        episode.watchdog_triggered = True
        if qualified_hook_current:
            episode.watchdog_rearmed = True
            episode.hook_action_opportunity_created = True
            episode.watchdog_phase = HookWatchdogPhase.REARM_GRACE.value
            episode.watchdog_rearmed_at = now
            episode.watchdog_deadline = min(
                now + self.rearm_grace_seconds,
                episode.hard_liveness_deadline
                if episode.hard_liveness_deadline is not None
                else now + self.rearm_grace_seconds,
            )
            return HookStallDecision(
                "rearm",
                "qualified_current_hook_evidence",
                float(state_age_seconds),
                hook_evidence_confidence,
                hook_evidence_age_seconds,
            )
        return self._sync_decision(
            reason="hook_action_stall_without_current_hook_evidence",
            state_age_seconds=state_age_seconds,
            hook_evidence_confidence=hook_evidence_confidence,
            hook_evidence_age_seconds=hook_evidence_age_seconds,
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
            "watchdog_phase": HookWatchdogPhase.NORMAL.value,
            "watchdog_rearmed_at": None,
            "watchdog_deadline": None,
            "hard_liveness_deadline": None,
            "terminal_outcome": None,
        }

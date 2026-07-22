"""One-shot Live CAST scheduling with visual acknowledgement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from src.fishing_v2.domain.observations import (
    GetObservation,
    PromptObservationKind,
    ResultBannerObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.ports.action_sink import ActionExecutionResult
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode


@dataclass(frozen=True)
class CastOpportunityConfig:
    visual_ack_timeout_seconds: float = 4.0
    clearance_freshness_seconds: float = 4.0
    stable_idle_frames_required: int = 2

    def __post_init__(self) -> None:
        if self.visual_ack_timeout_seconds <= 0:
            raise ValueError("cast visual acknowledgement timeout must be positive")
        if self.clearance_freshness_seconds <= 0:
            raise ValueError("post-cycle clearance freshness must be positive")
        if self.stable_idle_frames_required < 1:
            raise ValueError("stable idle frames must be positive")


class PresenceState(str, Enum):
    UNKNOWN = "unknown"
    ABSENT = "absent"
    PRESENT = "present"


class CastTerminalOutcome(str, Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    EMISSION_FAILED = "emission_failed"


class CastBlocker(str, Enum):
    RUNTIME_NOT_IDLE = "runtime_not_idle"
    PROMPT_NOT_IDLE_CAST = "prompt_not_idle_cast"
    IDLE_NOT_STABLE = "idle_not_stable"
    GET_PRESENCE_UNKNOWN = "get_presence_unknown"
    GET_PRESENT = "get_present"
    GET_ABSENCE_STALE = "get_absence_stale"
    RESULT_BANNER_UNKNOWN = "result_banner_unknown"
    RESULT_BANNER_PRESENT = "result_banner_present"
    RESULT_BANNER_ABSENCE_STALE = "result_banner_absence_stale"
    PHYSICAL_GET_EPISODE_ACTIVE = "physical_get_episode_active"
    COLLECT_NOT_COMPLETED = "collect_not_completed"
    POST_CYCLE_CLEARANCE_MISSING = "post_cycle_clearance_missing"
    OPPORTUNITY_ALREADY_CONSUMED = "opportunity_already_consumed"
    CAST_WAITING_ACK = "cast_waiting_ack"
    PANIC_LATCHED = "panic_latched"
    FOREGROUND_UNAVAILABLE = "foreground_unavailable"
    FOREGROUND_NOT_CONFIRMED = "foreground_not_confirmed"
    SAFETY_NOT_READY = "safety_not_ready"


@dataclass(frozen=True)
class PostCycleClearance:
    clearance_id: str
    source: str
    source_state: RuntimeState
    created_at: float
    expires_at: float
    cycle_id: str | None = None
    get_episode_id: str | None = None
    get_absence_certified: bool = False
    result_banner_absence_certified: bool = False
    consumed: bool = False
    terminal: bool = False


@dataclass(frozen=True)
class PostCycleClearanceStatus:
    tracking: bool
    stable_idle_frames: int
    get_presence_state: PresenceState
    get_evidence_age_ms: float | None
    get_absence_source: str | None
    result_banner_presence_state: PresenceState
    result_banner_evidence_age_ms: float | None
    clearance_id: str | None
    clearance_available: bool
    clearance_consumed: bool
    clearance_expired: bool


class PostCycleClearanceTracker:
    """Issue one short-lived certificate after either safe result path."""

    def __init__(self, config: CastOpportunityConfig | None = None) -> None:
        self.config = config or CastOpportunityConfig()
        self._sequence = 0
        self._tracking = False
        self._stable_idle_frames = 0
        self._last_prompt_frame_index: int | None = None
        self._get_presence = PresenceState.UNKNOWN
        self._get_evidence_at: float | None = None
        self._get_absence_source: str | None = None
        self._qualified_get_seen = False
        self._get_episode_id: str | None = None
        self._collect_visual_acknowledged = False
        self._collect_terminal_reason: str | None = None
        self._runtime_cycle_id: str | None = None
        self._banner_presence = PresenceState.UNKNOWN
        self._banner_evidence_at: float | None = None
        self._clearance: PostCycleClearance | None = None
        self._clearance_expired = False
        self._clearance_count = 0
        self._clearance_counts_by_source: dict[str, int] = {}

    def _start_result_window(self) -> None:
        self._tracking = True
        self._stable_idle_frames = 0
        self._last_prompt_frame_index = None
        self._get_presence = PresenceState.UNKNOWN
        self._get_evidence_at = None
        self._get_absence_source = None
        self._qualified_get_seen = False
        self._get_episode_id = None
        self._collect_visual_acknowledged = False
        self._collect_terminal_reason = None
        self._runtime_cycle_id = None
        self._banner_presence = PresenceState.UNKNOWN
        self._banner_evidence_at = None
        self._clearance = None
        self._clearance_expired = False

    @staticmethod
    def _age_ms(timestamp: float, observed_at: float | None) -> float | None:
        if observed_at is None:
            return None
        return max(0.0, (float(timestamp) - observed_at) * 1000.0)

    def observe(
        self,
        *,
        timestamp: float,
        previous_state: RuntimeState,
        current_state: RuntimeState,
        prompt_kind: PromptObservationKind | None,
        prompt_frame_index: int | None,
        get_observation: GetObservation | None,
        get_activation_mode: DetectorActivationMode,
        result_banner: ResultBannerObservation | None,
        physical_get_episode_open: bool,
        physical_get_episode_id: str | None = None,
        physical_get_episode_terminal: bool = False,
        physical_get_panel_visible: bool = False,
        collect_visual_acknowledged: bool = False,
        collect_complete_emission_count: int = 0,
        collect_terminal_reason: str | None = None,
        runtime_cycle_id: str | None = None,
        result_banner_hold_expired: bool = False,
    ) -> tuple[CastOpportunityEvent, ...]:
        events: list[CastOpportunityEvent] = []
        if (
            current_state == RuntimeState.RESULT_PENDING
            and previous_state != RuntimeState.RESULT_PENDING
        ) or (
            not self._tracking
            and self._clearance is None
            and previous_state == RuntimeState.RESULT_PENDING
            and current_state in {RuntimeState.RESULT_PENDING, RuntimeState.IDLE}
        ):
            self._start_result_window()

        if (
            self._clearance is not None
            and not self._clearance.consumed
            and not self._clearance_expired
        ):
            if float(timestamp) > self._clearance.expires_at:
                self._clearance_expired = True
                self._clearance = replace(self._clearance, terminal=True)
                payload = {
                    "clearance_id": self._clearance.clearance_id,
                    "clearance_source": self._clearance.source,
                    "created_at": self._clearance.created_at,
                    "expires_at": self._clearance.expires_at,
                }
                events.append(CastOpportunityEvent(
                    "post_cycle_clearance_expired", payload
                ))
                if self._clearance.source == "no_get_result":
                    events.append(CastOpportunityEvent(
                        "no_get_clearance_expired", payload
                    ))

        if not self._tracking:
            return tuple(events)

        if (
            prompt_frame_index is not None
            and prompt_frame_index != self._last_prompt_frame_index
        ):
            if prompt_kind == PromptObservationKind.IDLE_CAST:
                self._stable_idle_frames += 1
            else:
                self._stable_idle_frames = 0
            self._last_prompt_frame_index = prompt_frame_index

        if get_observation is not None and (
            get_observation.detected
            or get_activation_mode == DetectorActivationMode.BURST
        ):
            self._get_presence = (
                PresenceState.PRESENT
                if get_observation.detected else PresenceState.ABSENT
            )
            self._get_evidence_at = float(timestamp)
            if get_observation.detected:
                self._qualified_get_seen = True
                self._get_absence_source = None
            else:
                self._get_absence_source = get_observation.source

        if runtime_cycle_id is not None:
            self._runtime_cycle_id = runtime_cycle_id
        if physical_get_episode_id is not None:
            self._get_episode_id = physical_get_episode_id

        collected_get_complete = bool(
            self._qualified_get_seen
            and not self._collect_visual_acknowledged
            and self._get_episode_id is not None
            and physical_get_episode_terminal
            and not physical_get_episode_open
            and not physical_get_panel_visible
            and collect_visual_acknowledged
            and collect_complete_emission_count > 0
            and collect_terminal_reason
            == "qualified_get_panel_stably_disappeared"
        )
        if collected_get_complete:
            # This is a fresh visual certificate produced by the completed
            # physical episode. Detector OFF/None never reaches this branch.
            self._collect_visual_acknowledged = True
            self._collect_terminal_reason = collect_terminal_reason
            self._get_presence = PresenceState.ABSENT
            self._get_evidence_at = float(timestamp)
            self._get_absence_source = "post_collect_visual_ack"

        if result_banner is not None:
            self._banner_presence = (
                PresenceState.PRESENT
                if result_banner.detected else PresenceState.ABSENT
            )
            self._banner_evidence_at = float(timestamp)

        if physical_get_episode_open or physical_get_panel_visible:
            self._clearance = None
            return tuple(events)

        source = self._eligible_source(
            timestamp,
            result_banner_hold_expired=result_banner_hold_expired,
        )
        if current_state == RuntimeState.IDLE and source is not None:
            self._sequence += 1
            clearance_id = (
                f"no_get_clearance:{self._sequence}"
                if source == "no_get_result"
                else f"post_collect_clearance:{self._sequence}"
            )
            self._clearance = PostCycleClearance(
                clearance_id=clearance_id,
                source=source,
                source_state=RuntimeState.RESULT_PENDING,
                created_at=float(timestamp),
                expires_at=(
                    float(timestamp) + self.config.clearance_freshness_seconds
                ),
                cycle_id=self._runtime_cycle_id,
                get_episode_id=self._get_episode_id,
                get_absence_certified=True,
                result_banner_absence_certified=True,
            )
            self._clearance_expired = False
            self._clearance_count += 1
            self._clearance_counts_by_source[source] = (
                self._clearance_counts_by_source.get(source, 0) + 1
            )
            self._tracking = False
            payload = {
                "clearance_id": self._clearance.clearance_id,
                "clearance_source": source,
                "source_state": self._clearance.source_state.value,
                "created_at": self._clearance.created_at,
                "expires_at": self._clearance.expires_at,
                "cycle_id": self._clearance.cycle_id,
                "get_episode_id": self._clearance.get_episode_id,
                "get_absence_certified": True,
                "result_banner_absence_certified": True,
                "stable_idle_frames": self._stable_idle_frames,
                "get_presence_state": self._get_presence.value,
                "get_absence_source": self._get_absence_source,
                "result_banner_presence_state": self._banner_presence.value,
            }
            events.append(CastOpportunityEvent(
                "post_cycle_clearance_created", payload
            ))
            if source == "no_get_result":
                # Kept for existing session tooling; this is the same
                # clearance, not a second certificate.
                events.append(CastOpportunityEvent(
                    "no_get_clearance_created", payload
                ))
        elif current_state not in {RuntimeState.RESULT_PENDING, RuntimeState.IDLE}:
            # A qualified GET legitimately traverses GET/COLLECT_PENDING before
            # returning to IDLE. Preserve that physical episode identity.
            if not self._qualified_get_seen:
                self._tracking = False
        return tuple(events)

    def _eligible_source(
        self,
        timestamp: float,
        *,
        result_banner_hold_expired: bool,
    ) -> str | None:
        freshness_ms = self.config.clearance_freshness_seconds * 1000.0
        common = bool(
            self._stable_idle_frames >= self.config.stable_idle_frames_required
            and self._get_presence == PresenceState.ABSENT
            and self._age_ms(timestamp, self._get_evidence_at) is not None
            and self._age_ms(timestamp, self._get_evidence_at) <= freshness_ms
            and self._banner_presence != PresenceState.PRESENT
        )
        if not common:
            return None
        if self._qualified_get_seen:
            if (
                self._collect_visual_acknowledged
                and self._collect_terminal_reason
                == "qualified_get_panel_stably_disappeared"
                and (
                    self._banner_presence == PresenceState.ABSENT
                    or result_banner_hold_expired
                )
            ):
                return "collected_get_visual_ack"
            return None
        no_get_ready = bool(
            not self._qualified_get_seen
            and self._banner_presence == PresenceState.ABSENT
            and self._age_ms(timestamp, self._banner_evidence_at) is not None
            and self._age_ms(timestamp, self._banner_evidence_at) <= freshness_ms
        )
        return "no_get_result" if no_get_ready else None

    def current(self, timestamp: float) -> PostCycleClearance | None:
        clearance = self._clearance
        if (
            clearance is None
            or clearance.consumed
            or float(timestamp) > clearance.expires_at
        ):
            return None
        return clearance

    def consume(self, clearance_id: str) -> bool:
        if (
            self._clearance is None
            or self._clearance.clearance_id != clearance_id
            or self._clearance.consumed
        ):
            return False
        self._clearance = replace(
            self._clearance, consumed=True, terminal=True
        )
        return True

    def consume_with_events(
        self, clearance_id: str
    ) -> tuple[bool, tuple[CastOpportunityEvent, ...]]:
        if not self.consume(clearance_id):
            return False, ()
        clearance = self._clearance
        assert clearance is not None
        return True, (CastOpportunityEvent("post_cycle_clearance_consumed", {
            "clearance_id": clearance.clearance_id,
            "clearance_source": clearance.source,
            "cycle_id": clearance.cycle_id,
            "get_episode_id": clearance.get_episode_id,
            "consumed": True,
            "terminal": True,
        }),)

    def certified_get_absence(
        self, *, timestamp: float, frame_index: int
    ) -> GetObservation | None:
        clearance = self.current(timestamp)
        if clearance is None:
            return None
        return GetObservation(
            detected=False,
            confidence=1.0,
            frame_index=frame_index,
            timestamp=float(timestamp),
            source=(
                "post_collect_visual_ack"
                if clearance.source == "collected_get_visual_ack"
                else "post_cycle_no_get_clearance"
            ),
            evidence={
                "clearance_id": clearance.clearance_id,
                "source_state": clearance.source_state.value,
                "clearance_source": clearance.source,
                "expires_at": clearance.expires_at,
                "certified_absence": True,
            },
        )

    def status(self, timestamp: float) -> PostCycleClearanceStatus:
        clearance = self._clearance
        available = self.current(timestamp) is not None
        expired = bool(
            clearance is not None
            and not clearance.consumed
            and float(timestamp) > clearance.expires_at
        )
        return PostCycleClearanceStatus(
            tracking=self._tracking,
            stable_idle_frames=self._stable_idle_frames,
            get_presence_state=self._get_presence,
            get_evidence_age_ms=self._age_ms(timestamp, self._get_evidence_at),
            get_absence_source=self._get_absence_source,
            result_banner_presence_state=self._banner_presence,
            result_banner_evidence_age_ms=self._age_ms(
                timestamp, self._banner_evidence_at
            ),
            clearance_id=clearance.clearance_id if clearance else None,
            clearance_available=available,
            clearance_consumed=bool(clearance and clearance.consumed),
            clearance_expired=expired or self._clearance_expired,
        )

    def summary(self) -> dict[str, Any]:
        counts = dict(self._clearance_counts_by_source)
        return {
            "post_cycle_clearance_count": self._clearance_count,
            "post_cycle_clearance_counts_by_source": counts,
            "post_collect_clearance_count": counts.get(
                "collected_get_visual_ack", 0
            ),
            # Backwards-compatible summary field.
            "no_get_clearance_count": counts.get("no_get_result", 0),
        }


@dataclass(frozen=True)
class CastAttempt:
    opportunity_id: str
    action_id: str
    scheduled_at: float
    clearance_id: str


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
        self._terminal_outcome = CastTerminalOutcome.PENDING
        self._deadline: float | None = None
        self._emission_completed_at: float | None = None
        self._last_prompt_frame_index: int | None = None
        self._waiting_support_frames = 0
        self._late_ack_recorded = False
        self._opportunity_count = 0
        self._attempt_count = 0
        self._acknowledged_count = 0
        self._timeout_count = 0
        self._cancelled_count = 0
        self._emission_failed_count = 0
        self._late_ack_observed_count = 0
        self._terminal_outcome_counts: dict[str, int] = {}

    @property
    def opportunity_id(self) -> str | None:
        return self._opportunity_id

    @property
    def waiting_for_acknowledgement(self) -> bool:
        return self._open and self._input_completed and not self._terminal

    @property
    def opportunity_open(self) -> bool:
        return self._open

    @property
    def terminal_outcome(self) -> CastTerminalOutcome:
        return self._terminal_outcome

    @property
    def visual_ack_deadline(self) -> float | None:
        return self._deadline

    def schedule(
        self,
        *,
        timestamp: float,
        clearance_id: str | None,
        runtime_state: RuntimeState,
        prompt_kind: PromptObservationKind | None,
        physical_get_episode_open: bool,
    ) -> tuple[CastAttempt | None, tuple[CastOpportunityEvent, ...]]:
        if (
            clearance_id is None
            or runtime_state != RuntimeState.IDLE
            or prompt_kind != PromptObservationKind.IDLE_CAST
            or physical_get_episode_open
        ):
            return None, ()
        if self._open:
            return None, ()
        self._sequence += 1
        self._opportunity_id = f"cast_opportunity:{self._sequence}"
        self._open = True
        self._attempted = True
        self._input_completed = False
        self._terminal = False
        self._terminal_outcome = CastTerminalOutcome.PENDING
        self._deadline = None
        self._emission_completed_at = None
        self._last_prompt_frame_index = None
        self._waiting_support_frames = 0
        self._late_ack_recorded = False
        self._opportunity_count += 1
        self._attempt_count += 1
        attempt = CastAttempt(
            opportunity_id=self._opportunity_id,
            action_id=f"{self._opportunity_id}:CAST",
            scheduled_at=float(timestamp),
            clearance_id=clearance_id,
        )
        return attempt, (CastOpportunityEvent("cast_opportunity_started", {
            "opportunity_id": self._opportunity_id,
            "action_id": attempt.action_id,
            "clearance_id": clearance_id,
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
        if (
            not self._open
            or self._terminal
            or self._input_completed
            or attempt.opportunity_id != self._opportunity_id
        ):
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
            self._emission_completed_at = float(timestamp)
            self._deadline = float(timestamp) + self.config.visual_ack_timeout_seconds
        else:
            # A rejected, zero-event, or partial attempt is terminal. CAST never
            # retries automatically because a partial physical input is ambiguous.
            self._terminal = True
            self._terminal_outcome = CastTerminalOutcome.EMISSION_FAILED
            self._emission_failed_count += 1
            self._count_terminal(CastTerminalOutcome.EMISSION_FAILED)
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
        prompt_frame_index: int | None = None,
        prompt_timestamp: float | None = None,
    ) -> tuple[CastOpportunityEvent, ...]:
        if not self._open or not self._input_completed:
            return ()

        evidence_at = (
            float(prompt_timestamp) if prompt_timestamp is not None else None
        )
        is_new_prompt_frame = bool(
            prompt_frame_index is not None
            and prompt_frame_index != self._last_prompt_frame_index
        )
        waiting_after_emission = bool(
            prompt_kind == PromptObservationKind.WAITING_IN_PROGRESS
            and evidence_at is not None
            and self._emission_completed_at is not None
            and evidence_at > self._emission_completed_at
        )
        if is_new_prompt_frame:
            self._last_prompt_frame_index = prompt_frame_index
            if waiting_after_emission:
                self._waiting_support_frames += 1
            else:
                self._waiting_support_frames = 0

        waiting_stable = bool(
            waiting_after_emission
            and self._waiting_support_frames
            >= self.config.stable_idle_frames_required
        )
        evidence_before_deadline = bool(
            evidence_at is not None
            and self._deadline is not None
            and evidence_at <= self._deadline
        )

        if (
            not self._terminal
            and waiting_stable
            and evidence_before_deadline
        ):
            self._open = False
            self._terminal = True
            self._terminal_outcome = CastTerminalOutcome.ACKNOWLEDGED
            self._acknowledged_count += 1
            self._count_terminal(CastTerminalOutcome.ACKNOWLEDGED)
            return (CastOpportunityEvent("cast_visual_acknowledged", {
                "opportunity_id": self._opportunity_id,
                "cast_visual_acknowledged": True,
                "visual_acknowledgement": PromptObservationKind.WAITING_IN_PROGRESS.value,
                "visual_ack_evidence_at": evidence_at,
                "emission_completed_at": self._emission_completed_at,
                "visual_ack_deadline": self._deadline,
                "support_frames": self._waiting_support_frames,
                "runtime_state_before_ack": runtime_state.value,
                "terminal_outcome": CastTerminalOutcome.ACKNOWLEDGED.value,
                "os_input_emitted": False,
            }),)

        if (
            self._terminal_outcome == CastTerminalOutcome.TIMEOUT
            and waiting_stable
            and not self._late_ack_recorded
        ):
            self._late_ack_recorded = True
            self._late_ack_observed_count += 1
            return (CastOpportunityEvent("cast_late_ack_observed", {
                "opportunity_id": self._opportunity_id,
                "visual_ack_evidence_at": evidence_at,
                "visual_ack_deadline": self._deadline,
                "terminal_outcome": CastTerminalOutcome.TIMEOUT.value,
                "terminal_outcome_changed": False,
                "os_input_emitted": False,
            }),)
        if (
            not self._terminal
            and self._deadline is not None
            and float(timestamp) >= self._deadline
        ):
            self._terminal = True
            self._terminal_outcome = CastTerminalOutcome.TIMEOUT
            self._timeout_count += 1
            self._count_terminal(CastTerminalOutcome.TIMEOUT)
            return (CastOpportunityEvent("cast_visual_timeout", {
                "opportunity_id": self._opportunity_id,
                "cast_visual_acknowledged": False,
                "visual_ack_deadline": self._deadline,
                "terminal_outcome": CastTerminalOutcome.TIMEOUT.value,
                "retry_scheduled": False,
                "os_input_emitted": False,
            }),)
        return ()

    def cancel(self, *, timestamp: float, reason: str) -> tuple[CastOpportunityEvent, ...]:
        if not self._open or self._terminal:
            return ()
        self._terminal = True
        self._terminal_outcome = CastTerminalOutcome.CANCELLED
        self._cancelled_count += 1
        self._count_terminal(CastTerminalOutcome.CANCELLED)
        return (CastOpportunityEvent("cast_visual_cancelled", {
            "opportunity_id": self._opportunity_id,
            "timestamp": float(timestamp),
            "cancellation_reason": reason,
            "terminal_outcome": CastTerminalOutcome.CANCELLED.value,
            "retry_scheduled": False,
            "os_input_emitted": False,
        }),)

    def _count_terminal(self, outcome: CastTerminalOutcome) -> None:
        self._terminal_outcome_counts[outcome.value] = (
            self._terminal_outcome_counts.get(outcome.value, 0) + 1
        )

    def summary(self) -> dict[str, Any]:
        pending_count = max(
            0,
            self._opportunity_count - sum(self._terminal_outcome_counts.values()),
        )
        return {
            "cast_opportunity_count": self._opportunity_count,
            "cast_attempt_count": self._attempt_count,
            "cast_visual_acknowledged_count": self._acknowledged_count,
            "cast_timeout_count": self._timeout_count,
            "cast_cancelled_count": self._cancelled_count,
            "cast_emission_failed_count": self._emission_failed_count,
            "cast_pending_count": pending_count,
            "cast_late_ack_observed_count": self._late_ack_observed_count,
            "cast_terminal_outcome_counts": dict(self._terminal_outcome_counts),
        }

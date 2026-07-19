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
    source_state: RuntimeState
    created_at: float
    expires_at: float
    consumed: bool = False


@dataclass(frozen=True)
class PostCycleClearanceStatus:
    tracking: bool
    stable_idle_frames: int
    get_presence_state: PresenceState
    get_evidence_age_ms: float | None
    result_banner_presence_state: PresenceState
    result_banner_evidence_age_ms: float | None
    clearance_id: str | None
    clearance_available: bool
    clearance_consumed: bool
    clearance_expired: bool


class PostCycleClearanceTracker:
    """Issue one short-lived no-GET certificate from RESULT_PENDING evidence."""

    def __init__(self, config: CastOpportunityConfig | None = None) -> None:
        self.config = config or CastOpportunityConfig()
        self._sequence = 0
        self._tracking = False
        self._stable_idle_frames = 0
        self._last_prompt_frame_index: int | None = None
        self._get_presence = PresenceState.UNKNOWN
        self._get_evidence_at: float | None = None
        self._qualified_get_seen = False
        self._banner_presence = PresenceState.UNKNOWN
        self._banner_evidence_at: float | None = None
        self._clearance: PostCycleClearance | None = None
        self._clearance_expired = False
        self._clearance_count = 0

    def _start_result_window(self) -> None:
        self._tracking = True
        self._stable_idle_frames = 0
        self._last_prompt_frame_index = None
        self._get_presence = PresenceState.UNKNOWN
        self._get_evidence_at = None
        self._qualified_get_seen = False
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
                events.append(CastOpportunityEvent("no_get_clearance_expired", {
                    "clearance_id": self._clearance.clearance_id,
                    "created_at": self._clearance.created_at,
                    "expires_at": self._clearance.expires_at,
                }))

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

        if (
            get_observation is not None
            and get_activation_mode == DetectorActivationMode.BURST
        ):
            self._get_presence = (
                PresenceState.PRESENT
                if get_observation.detected else PresenceState.ABSENT
            )
            self._get_evidence_at = float(timestamp)
            if get_observation.detected:
                self._qualified_get_seen = True

        if result_banner is not None:
            self._banner_presence = (
                PresenceState.PRESENT
                if result_banner.detected else PresenceState.ABSENT
            )
            self._banner_evidence_at = float(timestamp)

        if self._qualified_get_seen or physical_get_episode_open:
            self._clearance = None
            if current_state not in {RuntimeState.RESULT_PENDING, RuntimeState.IDLE}:
                self._tracking = False
            return tuple(events)

        if current_state == RuntimeState.IDLE and self._can_create(timestamp):
            self._sequence += 1
            self._clearance = PostCycleClearance(
                clearance_id=f"no_get_clearance:{self._sequence}",
                source_state=RuntimeState.RESULT_PENDING,
                created_at=float(timestamp),
                expires_at=(
                    float(timestamp) + self.config.clearance_freshness_seconds
                ),
            )
            self._clearance_expired = False
            self._clearance_count += 1
            self._tracking = False
            events.append(CastOpportunityEvent("no_get_clearance_created", {
                "clearance_id": self._clearance.clearance_id,
                "source_state": self._clearance.source_state.value,
                "created_at": self._clearance.created_at,
                "expires_at": self._clearance.expires_at,
                "stable_idle_frames": self._stable_idle_frames,
                "get_presence_state": self._get_presence.value,
                "result_banner_presence_state": self._banner_presence.value,
            }))
        elif current_state not in {RuntimeState.RESULT_PENDING, RuntimeState.IDLE}:
            self._tracking = False
        return tuple(events)

    def _can_create(self, timestamp: float) -> bool:
        freshness_ms = self.config.clearance_freshness_seconds * 1000.0
        return bool(
            not self._qualified_get_seen
            and self._stable_idle_frames >= self.config.stable_idle_frames_required
            and self._get_presence == PresenceState.ABSENT
            and self._age_ms(timestamp, self._get_evidence_at) is not None
            and self._age_ms(timestamp, self._get_evidence_at) <= freshness_ms
            and self._banner_presence == PresenceState.ABSENT
            and self._age_ms(timestamp, self._banner_evidence_at) is not None
            and self._age_ms(timestamp, self._banner_evidence_at) <= freshness_ms
        )

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
        self._clearance = replace(self._clearance, consumed=True)
        return True

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
            source="post_cycle_no_get_clearance",
            evidence={
                "clearance_id": clearance.clearance_id,
                "source_state": clearance.source_state.value,
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
        return {"no_get_clearance_count": self._clearance_count}


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
        self._deadline: float | None = None
        self._opportunity_count = 0
        self._attempt_count = 0
        self._acknowledged_count = 0
        self._timeout_count = 0

    @property
    def opportunity_id(self) -> str | None:
        return self._opportunity_id

    @property
    def waiting_for_acknowledgement(self) -> bool:
        return self._open and self._input_completed and not self._terminal

    @property
    def opportunity_open(self) -> bool:
        return self._open

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
        self._deadline = None
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

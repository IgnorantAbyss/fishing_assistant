"""Stable IDLE visual certificates and unified Live CAST arming."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.fishing_v2.domain.observations import (
    PromptObservation,
    PromptObservationKind,
)


CAST_SOURCE_STARTUP = "startup_idle_confirmation"
CAST_SOURCE_RECOVERY = "authoritative_idle_recovery"
CAST_SOURCE_POST_CYCLE = "post_cycle_clearance"
CAST_SOURCE_TYPES = frozenset({
    CAST_SOURCE_STARTUP,
    CAST_SOURCE_RECOVERY,
    CAST_SOURCE_POST_CYCLE,
})


@dataclass(frozen=True)
class IdleRecoveryConfig:
    window_size: int = 5
    required_count: int = 4
    min_window_seconds: float = 0.5
    freshness_ms: float = 250.0
    cast_cooldown_seconds: float = 0.5
    cast_retry_min_interval_seconds: float = 3.0
    cast_liveness_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("idle recovery window size must be positive")
        if not 1 <= self.required_count <= self.window_size:
            raise ValueError("idle recovery required count must fit window")
        if self.min_window_seconds < 0:
            raise ValueError("idle recovery minimum window must be non-negative")
        if self.freshness_ms <= 0:
            raise ValueError("idle recovery freshness must be positive")
        if self.cast_cooldown_seconds < 0:
            raise ValueError("idle CAST cooldown must be non-negative")
        if self.cast_retry_min_interval_seconds <= 0:
            raise ValueError("idle CAST retry interval must be positive")
        if self.cast_liveness_timeout_seconds <= 0:
            raise ValueError("idle CAST liveness timeout must be positive")


@dataclass(frozen=True)
class IdleRecoveryCertificate:
    certificate_id: str
    physical_idle_id: str
    created_at: float
    window_start: float
    window_end: float
    observation_count: int
    idle_count: int
    latest_observation_age_ms: float

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdleRecoveryEvent:
    event_type: str
    payload: Mapping[str, Any]


class IdleRecoveryTracker:
    """Certify stable IDLE_CAST without treating missing detectors as absent."""

    def __init__(self, config: IdleRecoveryConfig | None = None) -> None:
        self.config = config or IdleRecoveryConfig()
        self._observations: deque[PromptObservation] = deque(
            maxlen=self.config.window_size
        )
        self._last_frame_index: int | None = None
        self._candidate_active = False
        self._episode_sequence = 0
        self._physical_idle_id: str | None = None
        self._certificate: IdleRecoveryCertificate | None = None

    @property
    def candidate_active(self) -> bool:
        return self._candidate_active

    @property
    def certificate(self) -> IdleRecoveryCertificate | None:
        return self._certificate

    def reset(self) -> None:
        self._observations.clear()
        self._last_frame_index = None
        self._candidate_active = False
        self._physical_idle_id = None
        self._certificate = None

    @staticmethod
    def _blocked(
        *,
        conflicting_evidence: bool,
        action_emission_in_progress: bool,
        key_currently_down: bool,
        panic_triggered: bool,
    ) -> str | None:
        if conflicting_evidence:
            return "qualified_specialized_or_result_evidence_present"
        if action_emission_in_progress:
            return "action_emission_in_progress"
        if key_currently_down:
            return "key_currently_down"
        if panic_triggered:
            return "panic_triggered"
        return None

    def observe(
        self,
        prompt: PromptObservation | None,
        *,
        timestamp: float,
        conflicting_evidence: bool,
        action_emission_in_progress: bool,
        key_currently_down: bool,
        panic_triggered: bool,
    ) -> tuple[IdleRecoveryCertificate | None, tuple[IdleRecoveryEvent, ...]]:
        events: list[IdleRecoveryEvent] = []
        blocker = self._blocked(
            conflicting_evidence=conflicting_evidence,
            action_emission_in_progress=action_emission_in_progress,
            key_currently_down=key_currently_down,
            panic_triggered=panic_triggered,
        )
        if blocker is not None:
            if self._candidate_active or self._certificate is not None:
                events.append(IdleRecoveryEvent(
                    "idle_recovery_candidate_cancelled",
                    {"reason": blocker},
                ))
            self.reset()
            return None, tuple(events)

        if (
            prompt is not None
            and prompt.frame_index != self._last_frame_index
        ):
            self._observations.append(prompt)
            self._last_frame_index = prompt.frame_index
            if (
                not self._candidate_active
                and prompt.kind == PromptObservationKind.IDLE_CAST
            ):
                self._episode_sequence += 1
                self._physical_idle_id = (
                    f"physical_idle:{self._episode_sequence}"
                )
                self._candidate_active = True
                events.append(IdleRecoveryEvent(
                    "idle_recovery_candidate_started",
                    {
                        "physical_idle_id": self._physical_idle_id,
                        "frame_index": prompt.frame_index,
                        "confidence": prompt.confidence,
                    },
                ))
            elif (
                self._certificate is not None
                and prompt.kind != PromptObservationKind.IDLE_CAST
            ):
                events.append(IdleRecoveryEvent(
                    "idle_recovery_candidate_cancelled",
                    {
                        "physical_idle_id": self._physical_idle_id,
                        "reason": "idle_prompt_disappeared",
                    },
                ))
                self.reset()
                return None, tuple(events)

        certificate = self._build_certificate(float(timestamp))
        if certificate is not None and self._certificate is None:
            self._certificate = certificate
        return self.current(
            timestamp=float(timestamp),
            conflicting_evidence=False,
            action_emission_in_progress=False,
            key_currently_down=False,
            panic_triggered=False,
        ), tuple(events)

    def _build_certificate(
        self, timestamp: float
    ) -> IdleRecoveryCertificate | None:
        if (
            not self._candidate_active
            or self._physical_idle_id is None
            or len(self._observations) < self.config.window_size
        ):
            return None
        values = tuple(self._observations)
        idle_count = sum(
            item.kind == PromptObservationKind.IDLE_CAST
            for item in values
        )
        window_start = float(values[0].timestamp)
        window_end = float(values[-1].timestamp)
        age_ms = max(0.0, (timestamp - window_end) * 1000.0)
        if (
            values[-1].kind != PromptObservationKind.IDLE_CAST
            or idle_count < self.config.required_count
            or window_end - window_start
            < self.config.min_window_seconds
            or age_ms > self.config.freshness_ms
        ):
            return None
        return IdleRecoveryCertificate(
            certificate_id=f"idle_certificate:{self._physical_idle_id}",
            physical_idle_id=self._physical_idle_id,
            created_at=timestamp,
            window_start=window_start,
            window_end=window_end,
            observation_count=len(values),
            idle_count=idle_count,
            latest_observation_age_ms=age_ms,
        )

    def current(
        self,
        *,
        timestamp: float,
        conflicting_evidence: bool,
        action_emission_in_progress: bool,
        key_currently_down: bool,
        panic_triggered: bool,
    ) -> IdleRecoveryCertificate | None:
        if self._blocked(
            conflicting_evidence=conflicting_evidence,
            action_emission_in_progress=action_emission_in_progress,
            key_currently_down=key_currently_down,
            panic_triggered=panic_triggered,
        ) is not None:
            return None
        candidate = self._build_certificate(float(timestamp))
        if candidate is None:
            return None
        if self._certificate is None:
            self._certificate = candidate
        return IdleRecoveryCertificate(
            **{
                **asdict(self._certificate),
                "latest_observation_age_ms": (
                    candidate.latest_observation_age_ms
                ),
            }
        )

    def summary(self, timestamp: float) -> dict[str, Any]:
        values = tuple(self._observations)
        return {
            "physical_idle_id": self._physical_idle_id,
            "observation_count": len(values),
            "idle_count": sum(
                item.kind == PromptObservationKind.IDLE_CAST
                for item in values
            ),
            "window_seconds": (
                float(values[-1].timestamp - values[0].timestamp)
                if len(values) >= 2 else 0.0
            ),
            "latest_observation_age_ms": (
                max(
                    0.0,
                    (float(timestamp) - values[-1].timestamp) * 1000.0,
                )
                if values else None
            ),
        }


@dataclass
class CastArmingRecord:
    source_type: str
    source_id: str
    physical_idle_id: str
    cycle_id: int
    armed_at: float
    eligible_at: float
    cast_started: bool = False
    emission_started: bool = False
    cast_applied: bool = False
    terminal_outcome: str | None = None
    consumed: bool = False
    opportunity_id: str | None = None
    merged_source_ids: list[str] = field(default_factory=list)

    def payload(self, timestamp: float) -> dict[str, Any]:
        return {
            **asdict(self),
            "cooldown_age_seconds": max(
                0.0, float(timestamp) - self.armed_at
            ),
        }


class CastArmingLifecycle:
    """Serialize every CAST source into one bounded physical-IDLE gate."""

    def __init__(self, *, retry_min_interval_seconds: float = 3.0) -> None:
        if retry_min_interval_seconds <= 0:
            raise ValueError("CAST retry minimum interval must be positive")
        self.retry_min_interval_seconds = float(
            retry_min_interval_seconds
        )
        self._active: CastArmingRecord | None = None
        self._seen_source_ids: set[str] = set()
        self._attempts_by_physical_idle: dict[str, int] = {}
        self._last_emission_started_at: float | None = None
        self._last_record: CastArmingRecord | None = None
        self._retry_authorized_physical_idle: set[str] = set()

    @property
    def active(self) -> CastArmingRecord | None:
        return self._active

    def recovery_physical_idle_id(self, default: str) -> str:
        if len(self._retry_authorized_physical_idle) == 1:
            return next(iter(self._retry_authorized_physical_idle))
        return default

    def request_cast_opportunity(
        self,
        *,
        source_type: str,
        source_id: str,
        physical_idle_id: str,
        cycle_id: int,
        timestamp: float,
        cooldown_seconds: float,
    ) -> tuple[CastArmingRecord | None, tuple[IdleRecoveryEvent, ...]]:
        if source_type not in CAST_SOURCE_TYPES:
            raise ValueError(f"Unsupported CAST source: {source_type}")
        if cooldown_seconds < 0:
            raise ValueError("CAST cooldown must be non-negative")
        dedupe_reason: str | None = None
        if source_id in self._seen_source_ids:
            dedupe_reason = "source_id_already_seen"
        elif self._active is not None:
            if (
                self._active.physical_idle_id == physical_idle_id
                and not self._active.cast_started
            ):
                self._seen_source_ids.add(source_id)
                self._active.merged_source_ids.append(source_id)
                dedupe_reason = "physical_idle_source_merged"
            else:
                dedupe_reason = "physical_idle_already_armed"
        else:
            attempts = self._attempts_by_physical_idle.get(
                physical_idle_id, 0
            )
            if attempts >= 2:
                dedupe_reason = "physical_idle_retry_limit_reached"
            elif attempts >= 1 and not (
                source_type == CAST_SOURCE_RECOVERY
                and physical_idle_id
                in self._retry_authorized_physical_idle
            ):
                dedupe_reason = "physical_idle_already_consumed"
        if dedupe_reason is not None:
            return None, (IdleRecoveryEvent(
                "cast_opportunity_deduplicated",
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "physical_idle_id": physical_idle_id,
                    "cycle_id": int(cycle_id),
                    "dedupe_reason": dedupe_reason,
                },
            ),)
        eligible_at = float(timestamp) + float(cooldown_seconds)
        if self._last_emission_started_at is not None:
            eligible_at = max(
                eligible_at,
                self._last_emission_started_at
                + self.retry_min_interval_seconds,
            )
        record = CastArmingRecord(
            source_type,
            source_id,
            physical_idle_id,
            int(cycle_id),
            float(timestamp),
            eligible_at,
        )
        self._seen_source_ids.add(source_id)
        if self._attempts_by_physical_idle.get(physical_idle_id, 0) >= 1:
            self._retry_authorized_physical_idle.discard(
                physical_idle_id
            )
        self._active = record
        return record, ()

    def authorize_retry_after_visual_timeout(self) -> str | None:
        record = self._last_record
        if (
            record is None
            or not record.consumed
            or self._attempts_by_physical_idle.get(
                record.physical_idle_id, 0
            ) != 1
        ):
            return None
        self._retry_authorized_physical_idle.add(
            record.physical_idle_id
        )
        return record.physical_idle_id

    def ready(
        self,
        *,
        timestamp: float,
        runtime_idle: bool,
        idle_certificate_valid: bool,
    ) -> bool:
        record = self._active
        return bool(
            record is not None
            and not record.cast_started
            and float(timestamp) >= record.eligible_at
            and runtime_idle
            and idle_certificate_valid
        )

    def cancel(self, reason: str) -> CastArmingRecord | None:
        record = self._active
        if record is None or record.cast_started:
            return None
        record.terminal_outcome = f"cancelled:{reason}"
        self._active = None
        return record

    def supersede(self, reason: str) -> CastArmingRecord | None:
        record = self._active
        if record is None or record.cast_started:
            return None
        record.terminal_outcome = f"superseded:{reason}"
        self._last_record = record
        self._active = None
        return record

    def expire_if_overdue(
        self,
        *,
        timestamp: float,
        service_timeout_seconds: float,
    ) -> CastArmingRecord | None:
        if service_timeout_seconds <= 0:
            raise ValueError("CAST arm service timeout must be positive")
        record = self._active
        if (
            record is None
            or record.cast_started
            or float(timestamp)
            < record.eligible_at + float(service_timeout_seconds)
        ):
            return None
        record.terminal_outcome = "expired:cast_arm_service_timeout"
        self._last_record = record
        self._active = None
        return record

    def mark_cast_started(
        self, opportunity_id: str
    ) -> CastArmingRecord | None:
        record = self._active
        if record is None or record.cast_started:
            return None
        record.cast_started = True
        record.opportunity_id = opportunity_id
        return record

    def record_execution(
        self,
        *,
        timestamp: float,
        emission_started: bool,
        applied: bool,
        terminal_outcome: str,
    ) -> CastArmingRecord | None:
        record = self._active
        if record is None:
            return None
        record.emission_started = bool(emission_started)
        record.cast_applied = bool(applied)
        record.consumed = bool(emission_started or applied)
        record.terminal_outcome = terminal_outcome
        if record.consumed:
            self._last_emission_started_at = float(timestamp)
            self._attempts_by_physical_idle[record.physical_idle_id] = (
                self._attempts_by_physical_idle.get(
                    record.physical_idle_id, 0
                ) + 1
            )
        self._last_record = record
        self._active = None
        return record

    def payload(self, timestamp: float) -> dict[str, Any]:
        return (
            self._active.payload(timestamp)
            if self._active is not None
            else {
                "source_type": None,
                "source_id": None,
                "cycle_id": None,
                "armed_at": None,
                "cast_started": False,
                "emission_started": False,
                "cast_applied": False,
                "terminal_outcome": None,
                "consumed": False,
            }
        )

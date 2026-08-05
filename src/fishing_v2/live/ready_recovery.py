"""Fresh, foreground-confirmed READY episode lifecycle.

This module does not emit input.  It only certifies that a stable READY prompt
was observed on new frames after foreground restoration and tracks whether the
physical READY opportunity has reached OS emission.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.fishing_v2.domain.observations import (
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState


@dataclass(frozen=True)
class ReadyRecoveryConfig:
    stable_frames: int
    min_confidence: float
    freshness_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within 0..1")
        if self.freshness_seconds <= 0.0:
            raise ValueError("freshness_seconds must be positive")


@dataclass(frozen=True)
class ReadyRecoveryCertificate:
    certificate_id: str
    physical_ready_episode_id: str
    created_at: float
    frame_index: int
    prompt_confidence: float
    prompt_age_seconds: float
    support_frames: int
    foreground_generation: int


@dataclass(frozen=True)
class ReadyRecoveryEvent:
    event_type: str
    reason: str


@dataclass(frozen=True)
class ReadyRecoveryUpdate:
    certificate: ReadyRecoveryCertificate | None
    events: tuple[ReadyRecoveryEvent, ...] = ()
    foreground_restored: bool = False


class ReadyRecoveryTracker:
    """Create one fresh certificate per physical READY presentation."""

    def __init__(self, config: ReadyRecoveryConfig) -> None:
        self.config = config
        self._foreground: bool | None = None
        self._foreground_generation = 0
        self._restore_frame_index: int | None = None
        self._support: deque[PromptObservation] = deque(
            maxlen=config.stable_frames
        )
        self._last_prompt_frame_index: int | None = None
        self._candidate_active = False
        self._episode_sequence = 0
        self._certificate_sequence = 0
        self._episode_id: str | None = None
        self._certificate: ReadyRecoveryCertificate | None = None
        self._awaiting_ready_clear = False
        self._emission_started = False
        self._emission_completed = False
        self._consumed = False
        self._terminal_without_retry = False

    @property
    def physical_ready_episode_id(self) -> str | None:
        return self._episode_id

    @property
    def emission_started(self) -> bool:
        return self._emission_started

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def proposal_allowed(self) -> bool:
        return not (
            self._consumed
            or self._terminal_without_retry
            or self._awaiting_ready_clear
        )

    def _clear_candidate(self) -> None:
        self._support.clear()
        self._candidate_active = False
        self._certificate = None

    def _end_episode(self) -> None:
        self._clear_candidate()
        self._episode_id = None
        self._emission_started = False
        self._emission_completed = False
        self._consumed = False
        self._terminal_without_retry = False
        self._awaiting_ready_clear = False

    def record_emission(
        self,
        *,
        started: bool,
        completed: bool,
        committed: bool,
    ) -> None:
        """Record the terminal START_HOOK result without initiating input."""
        if started:
            self._emission_started = True
        if completed:
            self._emission_completed = True
        if started and not (completed and committed):
            # A partial/failed emission is fail-closed for this physical panel.
            self._terminal_without_retry = True
            self._awaiting_ready_clear = True
        if completed and committed:
            self._consumed = True
            self._awaiting_ready_clear = True

    def observe(
        self,
        *,
        frame_index: int,
        timestamp: float,
        runtime_state: RuntimeState,
        prompt: PromptObservation | None,
        foreground: bool,
        target_valid: bool,
        conflicting_evidence: bool,
        panic_triggered: bool,
        action_emission_in_progress: bool,
    ) -> ReadyRecoveryUpdate:
        events: list[ReadyRecoveryEvent] = []
        restored = False
        if self._foreground is not False and not foreground:
            events.append(ReadyRecoveryEvent("foreground_lost", "target_not_foreground"))
            self._clear_candidate()
        elif self._foreground is False and foreground:
            self._foreground_generation += 1
            self._restore_frame_index = int(frame_index)
            restored = True
            events.append(ReadyRecoveryEvent("foreground_restored", "target_foreground_confirmed"))
            self._clear_candidate()
        self._foreground = bool(foreground)

        prompt_is_ready = bool(
            prompt is not None
            and prompt.kind == PromptObservationKind.READY_BITE
        )
        if not prompt_is_ready:
            if self._candidate_active:
                events.append(ReadyRecoveryEvent(
                    "ready_recovery_candidate_cancelled",
                    "ready_prompt_disappeared",
                ))
            if prompt is not None and prompt.kind != PromptObservationKind.READY_BITE:
                self._end_episode()
            return ReadyRecoveryUpdate(None, tuple(events), restored)

        assert prompt is not None
        if self._awaiting_ready_clear:
            return ReadyRecoveryUpdate(None, tuple(events), restored)

        prompt_age = max(0.0, float(timestamp) - float(prompt.timestamp))
        post_restore_frame = bool(
            self._restore_frame_index is None
            or prompt.frame_index > self._restore_frame_index
        )
        eligible = bool(
            prompt.confidence >= self.config.min_confidence
            and prompt_age <= self.config.freshness_seconds
            and foreground
            and target_valid
            and post_restore_frame
            and not conflicting_evidence
            and not panic_triggered
            and not action_emission_in_progress
        )
        if not eligible:
            if self._candidate_active:
                events.append(ReadyRecoveryEvent(
                    "ready_recovery_candidate_cancelled",
                    "ready_certificate_precondition_failed",
                ))
            self._clear_candidate()
            return ReadyRecoveryUpdate(None, tuple(events), restored)

        if prompt.frame_index != self._last_prompt_frame_index:
            self._last_prompt_frame_index = prompt.frame_index
            if not self._candidate_active:
                self._candidate_active = True
                events.append(ReadyRecoveryEvent(
                    "ready_recovery_candidate_started",
                    "fresh_ready_prompt_candidate",
                ))
            self._support.append(prompt)
        if len(self._support) < self.config.stable_frames:
            return ReadyRecoveryUpdate(None, tuple(events), restored)

        if self._episode_id is None:
            self._episode_sequence += 1
            self._episode_id = f"ready:{self._episode_sequence}"
        if self._certificate is None:
            self._certificate_sequence += 1
            self._certificate = ReadyRecoveryCertificate(
                certificate_id=f"ready-cert:{self._certificate_sequence}",
                physical_ready_episode_id=self._episode_id,
                created_at=float(timestamp),
                frame_index=int(frame_index),
                prompt_confidence=float(prompt.confidence),
                prompt_age_seconds=prompt_age,
                support_frames=len(self._support),
                foreground_generation=self._foreground_generation,
            )
            events.append(ReadyRecoveryEvent(
                "ready_recovery_certificate_created",
                "fresh_stable_ready_after_foreground_confirmation",
            ))
        return ReadyRecoveryUpdate(self._certificate, tuple(events), restored)

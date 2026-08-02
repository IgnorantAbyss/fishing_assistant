"""Configurable Prompt polling cadence for the finalized runtime flow."""

from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.observations import (
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState


@dataclass(frozen=True)
class PromptPollingConfig:
    waiting_interval_seconds: float = 4.0
    waiting_min_seconds: float = 3.0
    waiting_max_seconds: float = 5.0
    ready_fps: float = 20.0
    result_pending_fps: float = 5.0
    ready_confirmation_timeout_seconds: float = 0.3

    def __post_init__(self) -> None:
        if not self.waiting_min_seconds <= self.waiting_interval_seconds <= self.waiting_max_seconds:
            raise ValueError("WAITING Prompt interval must be inside its configured min/max range")
        if not (3.0 <= self.waiting_min_seconds <= self.waiting_max_seconds <= 5.0):
            raise ValueError("Reviewed WAITING Prompt range is 3..5 seconds")
        if self.ready_fps <= 0 or self.result_pending_fps <= 0:
            raise ValueError("Prompt polling FPS must be positive")
        if not 0.15 <= self.ready_confirmation_timeout_seconds <= 0.3:
            raise ValueError("READY confirmation timeout must be between 150 and 300 ms")


@dataclass(frozen=True)
class ReadyPromptBurstUpdate:
    burst_started: bool = False
    burst_cancelled: bool = False
    burst_timed_out: bool = False
    candidate_reset_required: bool = False
    ready_support_frames: int = 0
    candidate_age_seconds: float = 0.0
    reason: str = "no_ready_candidate"


@dataclass(frozen=True)
class MissedReadyRecoveryUpdate:
    candidate_started: bool = False
    candidate_cancelled: bool = False
    recovered: bool = False
    support_frames: int = 0
    candidate_age_seconds: float = 0.0
    reason: str = "no_hook_instruction_candidate"


class MissedReadyRecoveryTracker:
    """Confirm a threshold-qualified HOOK hint inside a short WAITING burst."""

    def __init__(
        self,
        *,
        stable_frames: int,
        min_confidence: float,
        confirmation_timeout_seconds: float,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within 0..1")
        if confirmation_timeout_seconds <= 0:
            raise ValueError("confirmation_timeout_seconds must be positive")
        self.stable_frames = int(stable_frames)
        self.min_confidence = float(min_confidence)
        self.confirmation_timeout_seconds = float(
            confirmation_timeout_seconds
        )
        self._started_at: float | None = None
        self._support_frames = 0
        self._last_frame_index: int | None = None

    @property
    def active(self) -> bool:
        return self._started_at is not None

    def reset(self) -> None:
        self._started_at = None
        self._support_frames = 0
        self._last_frame_index = None

    def observe(
        self,
        state: RuntimeState,
        prompt: PromptObservation | None,
    ) -> MissedReadyRecoveryUpdate:
        eligible = bool(
            state == RuntimeState.WAITING
            and prompt is not None
            and prompt.kind == PromptObservationKind.HOOK_INSTRUCTION
            and prompt.confidence >= self.min_confidence
        )
        if not eligible:
            cancelled = self.active
            self.reset()
            return MissedReadyRecoveryUpdate(
                candidate_cancelled=cancelled,
                reason=(
                    "runtime_not_waiting"
                    if state != RuntimeState.WAITING
                    else "hook_instruction_below_existing_confidence_gate"
                    if (
                        prompt is not None
                        and prompt.kind
                        == PromptObservationKind.HOOK_INSTRUCTION
                    )
                    else "hook_instruction_candidate_absent"
                ),
            )

        assert prompt is not None
        now = float(prompt.timestamp)
        timed_out = bool(
            self._started_at is not None
            and now - self._started_at
            > self.confirmation_timeout_seconds
        )
        if timed_out:
            self.reset()
        started = self._started_at is None
        if started:
            self._started_at = now
        if prompt.frame_index != self._last_frame_index:
            self._support_frames += 1
            self._last_frame_index = prompt.frame_index
        age = now - self._started_at
        if self._support_frames < self.stable_frames:
            return MissedReadyRecoveryUpdate(
                candidate_started=started,
                candidate_cancelled=timed_out,
                support_frames=self._support_frames,
                candidate_age_seconds=age,
                reason=(
                    "hook_instruction_candidate_restarted_after_timeout"
                    if timed_out
                    else "hook_instruction_candidate_pending"
                ),
            )
        support_frames = self._support_frames
        self.reset()
        return MissedReadyRecoveryUpdate(
            recovered=True,
            support_frames=support_frames,
            candidate_age_seconds=age,
            reason="stable_hook_instruction_while_waiting",
        )


class RuntimeSchedulePolicy:
    def __init__(self, config: PromptPollingConfig | None = None) -> None:
        self.config = config or PromptPollingConfig()
        self._ready_candidate_started_at: float | None = None
        self._ready_support_frames = 0

    @property
    def ready_candidate_active(self) -> bool:
        return self._ready_candidate_started_at is not None

    def reset(self) -> None:
        self._clear_ready_candidate()

    def _clear_ready_candidate(self) -> None:
        self._ready_candidate_started_at = None
        self._ready_support_frames = 0

    def observe_prompt(
        self,
        state: RuntimeState,
        prompt: PromptObservationKind,
        timestamp: float,
    ) -> ReadyPromptBurstUpdate:
        """Update the existing Prompt schedule after one real observation.

        READY_BITE is already the PromptObserver's threshold- and ambiguity-
        qualified output. Raw labels and specialized detector evidence never
        enter this scheduling decision.
        """
        if state != RuntimeState.WAITING:
            cancelled = self.ready_candidate_active
            self._clear_ready_candidate()
            return ReadyPromptBurstUpdate(
                burst_cancelled=cancelled,
                candidate_reset_required=cancelled,
                reason="runtime_not_waiting",
            )

        now = float(timestamp)
        timed_out = bool(
            self._ready_candidate_started_at is not None
            and now - self._ready_candidate_started_at
            > self.config.ready_confirmation_timeout_seconds
        )
        if timed_out:
            self._clear_ready_candidate()

        if prompt == PromptObservationKind.READY_BITE:
            started = self._ready_candidate_started_at is None
            if started:
                self._ready_candidate_started_at = now
            self._ready_support_frames += 1
            return ReadyPromptBurstUpdate(
                burst_started=started,
                burst_timed_out=timed_out,
                candidate_reset_required=timed_out,
                ready_support_frames=self._ready_support_frames,
                candidate_age_seconds=now - self._ready_candidate_started_at,
                reason=(
                    "ready_candidate_restarted_after_timeout"
                    if timed_out else "ready_candidate_burst"
                ),
            )

        cancelled = self.ready_candidate_active
        self._clear_ready_candidate()
        return ReadyPromptBurstUpdate(
            burst_cancelled=cancelled,
            burst_timed_out=timed_out,
            candidate_reset_required=cancelled or timed_out,
            reason=(
                f"ready_candidate_cancelled_by_{prompt.value.lower()}"
                if cancelled else "no_ready_candidate"
            ),
        )

    def confirm_ready(self, timestamp: float) -> ReadyPromptBurstUpdate:
        started_at = self._ready_candidate_started_at
        support_frames = self._ready_support_frames
        age = max(0.0, float(timestamp) - started_at) if started_at is not None else 0.0
        self._clear_ready_candidate()
        return ReadyPromptBurstUpdate(
            burst_cancelled=started_at is not None,
            ready_support_frames=support_frames,
            candidate_age_seconds=age,
            reason="ready_candidate_confirmed",
        )

    def prompt_interval_seconds(self, state: RuntimeState) -> float | None:
        if state == RuntimeState.WAITING:
            if self.ready_candidate_active:
                return 1.0 / self.config.ready_fps
            return self.config.waiting_interval_seconds
        if state == RuntimeState.READY:
            return 1.0 / self.config.ready_fps
        if state == RuntimeState.RESULT_PENDING:
            return 1.0 / self.config.result_pending_fps
        return None

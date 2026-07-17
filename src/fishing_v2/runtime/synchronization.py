"""Startup consensus without pretending an observer exists."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_evidence import QualifiedObservationBundle


@dataclass(frozen=True)
class SynchronizationConfig:
    observation_frames: int = 10
    minimum_consensus_ratio: float = 0.7
    minimum_confidence: float = 0.8
    timeout_sec: float = 5.0


@dataclass(frozen=True)
class SynchronizationResult:
    state: RuntimeState
    synchronized: bool
    confidence: float
    reason: str
    candidate_state: RuntimeState | None = None
    support_frames: int = 0
    observed_frames: int = 0
    duration_seconds: float = 0.0
    rejection_reason: str | None = None


class StartupSynchronizer:
    def __init__(self, config: SynchronizationConfig | None = None, *, started_at: float = 0.0) -> None:
        self.config = config or SynchronizationConfig()
        self.started_at = float(started_at)
        self._prompt_votes: list[tuple[RuntimeState, float]] = []
        self._frames = 0

    def reset(self, *, started_at: float) -> None:
        """Start a fresh evidence window without carrying stale votes."""
        self.started_at = float(started_at)
        self._prompt_votes.clear()
        self._frames = 0

    def observe_recovery(
        self,
        qualified: QualifiedObservationBundle,
        *,
        has_conflict: bool,
    ) -> SynchronizationResult:
        """Recover from SYNC_REQUIRED using a fresh, conflict-free window.

        Prompt-derived states use the same frame count, confidence and consensus
        thresholds as startup. Specialized states can recover immediately only
        after their detector evidence has passed the normal qualifier.
        """
        bundle = qualified.bundle
        duration = max(0.0, bundle.timestamp - self.started_at)
        if has_conflict:
            self.reset(started_at=bundle.timestamp)
            return SynchronizationResult(
                RuntimeState.SYNC_REQUIRED,
                False,
                0.0,
                "sync_recovery_conflicting_evidence",
                observed_frames=0,
                duration_seconds=0.0,
                rejection_reason="conflicting_evidence_reset_window",
            )

        self._frames += 1
        specials = (
            (bundle.get, qualified.get.qualified_detected, RuntimeState.GET),
            (bundle.press, qualified.press.qualified_detected, RuntimeState.PRESS),
            (bundle.hook, qualified.hook.qualified_detected, RuntimeState.HOOK),
        )
        for observation, is_qualified, state in specials:
            if (
                observation is not None
                and is_qualified
                and observation.detected
                and observation.confidence >= self.config.minimum_confidence
            ):
                return SynchronizationResult(
                    state,
                    True,
                    observation.confidence,
                    f"qualified_{state.value.lower()}_sync_recovery",
                    candidate_state=state,
                    support_frames=1,
                    observed_frames=self._frames,
                    duration_seconds=duration,
                )

        mapping = {
            PromptObservationKind.IDLE_CAST: RuntimeState.IDLE,
            PromptObservationKind.WAITING_IN_PROGRESS: RuntimeState.WAITING,
            PromptObservationKind.READY_BITE: RuntimeState.READY,
        }
        prompt = bundle.prompt
        candidate = mapping.get(prompt.kind) if prompt is not None else None
        rejection: str | None = None
        if candidate == RuntimeState.IDLE:
            get_absence_confirmed = bool(
                bundle.get is not None
                and qualified.get.activation_mode.value != "OFF"
                and not qualified.get.raw_detected
            )
            if not get_absence_confirmed:
                candidate = None
                rejection = "idle_requires_qualified_get_absence"
        if candidate is not None and prompt is not None:
            if prompt.confidence >= self.config.minimum_confidence:
                self._prompt_votes.append((candidate, prompt.confidence))
            else:
                rejection = "prompt_below_sync_confidence"
                candidate = None
        elif rejection is None:
            rejection = "unknown_or_non_state_prompt"

        counts = Counter(state for state, _ in self._prompt_votes)
        leading_state: RuntimeState | None = None
        leading_count = 0
        leading_confidence = 0.0
        if counts:
            leading_state, leading_count = counts.most_common(1)[0]
            leading_confidence = sum(
                value for vote, value in self._prompt_votes if vote == leading_state
            ) / leading_count

        complete = self._frames >= self.config.observation_frames
        timed_out = duration >= self.config.timeout_sec
        if complete or timed_out:
            ratio = leading_count / max(self._frames, 1)
            if leading_state is not None and ratio >= self.config.minimum_consensus_ratio:
                return SynchronizationResult(
                    leading_state,
                    True,
                    leading_confidence,
                    "prompt_consensus_sync_recovery",
                    candidate_state=leading_state,
                    support_frames=leading_count,
                    observed_frames=self._frames,
                    duration_seconds=duration,
                )
            result = SynchronizationResult(
                RuntimeState.SYNC_REQUIRED,
                False,
                leading_confidence,
                "sync_recovery_consensus_not_reached",
                candidate_state=leading_state,
                support_frames=leading_count,
                observed_frames=self._frames,
                duration_seconds=duration,
                rejection_reason=rejection or "insufficient_consensus",
            )
            self.reset(started_at=bundle.timestamp)
            return result

        return SynchronizationResult(
            RuntimeState.SYNC_REQUIRED,
            False,
            leading_confidence,
            "collecting_sync_recovery_observations",
            candidate_state=leading_state,
            support_frames=leading_count,
            observed_frames=self._frames,
            duration_seconds=duration,
            rejection_reason=rejection,
        )

    def observe(
        self,
        bundle: ObservationBundle,
        *,
        manual_override: RuntimeState | None = None,
    ) -> SynchronizationResult:
        if manual_override is not None:
            if manual_override in {RuntimeState.SYNCING, RuntimeState.SYNC_REQUIRED}:
                raise ValueError("Manual start-state override must be a concrete runtime state")
            return SynchronizationResult(manual_override, True, 1.0, "manual_start_state_override")
        self._frames += 1
        specials = (
            (bundle.get, RuntimeState.GET),
            (bundle.press, RuntimeState.PRESS),
            (bundle.hook, RuntimeState.HOOK),
        )
        for observation, state in specials:
            if observation and observation.detected and observation.confidence >= self.config.minimum_confidence:
                return SynchronizationResult(state, True, observation.confidence, f"strong_{state.value.lower()}_startup_evidence")
        mapping = {
            PromptObservationKind.IDLE_CAST: RuntimeState.IDLE,
            PromptObservationKind.WAITING_IN_PROGRESS: RuntimeState.WAITING,
            PromptObservationKind.READY_BITE: RuntimeState.READY,
        }
        if bundle.prompt and bundle.prompt.kind in mapping and bundle.prompt.confidence >= self.config.minimum_confidence:
            self._prompt_votes.append((mapping[bundle.prompt.kind], bundle.prompt.confidence))
        complete = self._frames >= self.config.observation_frames
        timed_out = bundle.timestamp - self.started_at >= self.config.timeout_sec
        if not (complete or timed_out):
            return SynchronizationResult(RuntimeState.SYNCING, False, 0.0, "collecting_startup_observations")
        counts = Counter(state for state, _ in self._prompt_votes)
        if counts:
            state, count = counts.most_common(1)[0]
            ratio = count / max(self._frames, 1)
            confidence = sum(value for vote, value in self._prompt_votes if vote == state) / count
            if ratio >= self.config.minimum_consensus_ratio:
                return SynchronizationResult(state, True, confidence, "prompt_consensus")
        return SynchronizationResult(RuntimeState.SYNC_REQUIRED, False, 0.0, "startup_consensus_not_reached")

"""Startup consensus without pretending an observer exists."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.perception.observation_bundle import ObservationBundle


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


class StartupSynchronizer:
    def __init__(self, config: SynchronizationConfig | None = None, *, started_at: float = 0.0) -> None:
        self.config = config or SynchronizationConfig()
        self.started_at = float(started_at)
        self._prompt_votes: list[tuple[RuntimeState, float]] = []
        self._frames = 0

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
            PromptObservationKind.IDLE_PROMPT: RuntimeState.IDLE,
            PromptObservationKind.WAITING_PROMPT: RuntimeState.WAITING,
            PromptObservationKind.READY_PROMPT: RuntimeState.READY,
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

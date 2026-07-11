"""Configurable Prompt polling cadence for the finalized runtime flow."""

from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.runtime_state import RuntimeState


@dataclass(frozen=True)
class PromptPollingConfig:
    waiting_interval_seconds: float = 4.0
    waiting_min_seconds: float = 3.0
    waiting_max_seconds: float = 5.0
    ready_fps: float = 5.0
    result_pending_fps: float = 5.0

    def __post_init__(self) -> None:
        if not self.waiting_min_seconds <= self.waiting_interval_seconds <= self.waiting_max_seconds:
            raise ValueError("WAITING Prompt interval must be inside its configured min/max range")
        if not (3.0 <= self.waiting_min_seconds <= self.waiting_max_seconds <= 5.0):
            raise ValueError("Reviewed WAITING Prompt range is 3..5 seconds")
        if self.ready_fps <= 0 or self.result_pending_fps <= 0:
            raise ValueError("Prompt polling FPS must be positive")


class RuntimeSchedulePolicy:
    def __init__(self, config: PromptPollingConfig | None = None) -> None:
        self.config = config or PromptPollingConfig()

    def prompt_interval_seconds(self, state: RuntimeState) -> float | None:
        if state == RuntimeState.WAITING:
            return self.config.waiting_interval_seconds
        if state == RuntimeState.READY:
            return 1.0 / self.config.ready_fps
        if state == RuntimeState.RESULT_PENDING:
            return 1.0 / self.config.result_pending_fps
        return None

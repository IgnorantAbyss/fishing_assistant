"""Conservative temporal smoothing for offline replay analysis only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ALLOWED_TRANSITIONS = {
    "IDLE": {"IDLE", "WAITING"},
    "WAITING": {"WAITING", "READY"},
    "READY": {"READY", "HOOK"},
    "HOOK": {"HOOK", "PRESS", "GET"},
    "PRESS": {"PRESS", "GET", "IDLE"},
    "GET": {"GET", "IDLE"},
}


@dataclass(frozen=True)
class StateSmoother:
    max_unknown_gap: int = 4
    ready_to_hook_gap: int = 4

    def smooth(self, states: Sequence[str], confidences: Sequence[float] | None = None) -> list[str]:
        """Fill only short, context-supported gaps; never rewrite a long UNKNOWN run."""
        smoothed = list(states)
        if not smoothed:
            return smoothed

        # An isolated blip surrounded by the same state is normally a detector
        # fluctuation, including a one-frame false special state.
        for index in range(1, len(smoothed) - 1):
            if (
                smoothed[index - 1] != "UNKNOWN"
                and smoothed[index - 1] == smoothed[index + 1]
                and smoothed[index] != smoothed[index - 1]
            ):
                smoothed[index] = smoothed[index - 1]

        # Fill bounded UNKNOWN gaps only when both endpoints provide evidence.
        index = 0
        while index < len(smoothed):
            if smoothed[index] != "UNKNOWN":
                index += 1
                continue
            end = index
            while end < len(smoothed) and smoothed[end] == "UNKNOWN":
                end += 1
            gap = end - index
            previous = smoothed[index - 1] if index else None
            following = smoothed[end] if end < len(smoothed) else None
            if gap <= self.max_unknown_gap and previous is not None and previous == following:
                smoothed[index:end] = [previous] * gap
            elif gap <= self.ready_to_hook_gap and previous == "READY" and following == "HOOK":
                # The hook UI can briefly be absent after READY before its bar
                # appears; this is a transition, not a broad UNKNOWN fallback.
                smoothed[index:end] = ["HOOK"] * gap
            elif gap <= self.max_unknown_gap and previous == "PRESS" and following == "GET":
                smoothed[index:end] = ["PRESS"] * gap
            index = end

        # If a short low-confidence segment immediately before a confirmed HOOK
        # follows READY, promote just that transition interval to HOOK.
        for index, state in enumerate(smoothed):
            if state != "HOOK":
                continue
            start = index
            while start > 0 and smoothed[start - 1] == "HOOK":
                start -= 1
            lookback = max(0, start - self.ready_to_hook_gap)
            ready_candidates = [
                prior for prior in range(max(0, lookback - 8), start) if smoothed[prior] == "READY"
            ]
            if not ready_candidates:
                continue
            transition_start = max(lookback, ready_candidates[-1] + 1)
            for prior in range(transition_start, start):
                if smoothed[prior] in {"UNKNOWN", "WAITING", "IDLE", "READY"}:
                    smoothed[prior] = "HOOK"
            break
        return smoothed

"""Latest-frame Hook ROI scheduling and per-episode timing telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HookROIFrame:
    frame_index: int
    timestamp: float
    pixels: np.ndarray


class LatestHookFrameSlot:
    """A capacity-one slot: publishing replaces any unconsumed stale frame."""

    def __init__(self) -> None:
        self._latest: HookROIFrame | None = None
        self.stale_frames_dropped = 0

    def publish(self, frame: HookROIFrame) -> None:
        if self._latest is not None:
            self.stale_frames_dropped += 1
        self._latest = frame

    def take_latest(self) -> HookROIFrame | None:
        frame = self._latest
        self._latest = None
        return frame


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class HookEpisodeTelemetry:
    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self._active is not None

    def start(self, timestamp: float, stale_frames_dropped: int) -> None:
        if self._active is not None:
            return
        self._active = {
            "episode_index": len(self._episodes) + 1,
            "started_at": float(timestamp),
            "timestamps": [],
            "stale_start": int(stale_frames_dropped),
        }

    def record_detector_frame(self, timestamp: float) -> None:
        if self._active is not None:
            self._active["timestamps"].append(float(timestamp))

    def finish(self, timestamp: float, stale_frames_dropped: int) -> None:
        if self._active is None:
            return
        active = self._active
        timestamps = list(active["timestamps"])
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(timestamps, timestamps[1:])
            if right >= left
        ]
        active_duration = max(
            0.0,
            float(timestamp) - float(active["started_at"]),
        )
        sampled_duration = (
            timestamps[-1] - timestamps[0]
            if len(timestamps) >= 2 else 0.0
        )
        actual_fps = (
            (len(timestamps) - 1) / sampled_duration
            if len(timestamps) >= 2 and sampled_duration > 0.0
            else 0.0
        )
        maximum = max(intervals, default=0.0)
        maximum_index = (
            intervals.index(maximum) if intervals else None
        )
        max_gap_diagnostic = None
        if maximum_index is not None and maximum > 45.0:
            max_gap_diagnostic = {
                "reason": "hook_frame_gap_above_45ms",
                "interval_ms": maximum,
                "from_timestamp": timestamps[maximum_index],
                "to_timestamp": timestamps[maximum_index + 1],
                "gap_count_above_45ms": sum(
                    interval > 45.0 for interval in intervals
                ),
            }
        self._episodes.append({
            "episode_index": active["episode_index"],
            "hook_detector_frame_count": len(timestamps),
            "hook_detector_active_duration": active_duration,
            "hook_detector_actual_fps": actual_fps,
            "hook_frame_interval_mean_ms": (
                sum(intervals) / len(intervals) if intervals else 0.0
            ),
            "hook_frame_interval_p95_ms": _percentile(intervals, 0.95),
            "hook_frame_interval_max_ms": maximum,
            "stale_hook_frames_dropped": (
                int(stale_frames_dropped) - int(active["stale_start"])
            ),
            "max_gap_diagnostic": max_gap_diagnostic,
        })
        self._active = None

    def summaries(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._episodes]


class HookCriticalFrameAssembler:
    """Places a precise ROI into the original coordinate system."""

    def __init__(
        self,
        frame_size: tuple[int, int],
        hook_bounds: tuple[int, int, int, int],
        prompt_bounds: tuple[int, int, int, int],
    ) -> None:
        width, height = frame_size
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.hook_bounds = hook_bounds
        self.prompt_bounds = prompt_bounds

    def update_prompt_context(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = self.prompt_bounds
        self._canvas[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    def compose(self, hook_roi: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.hook_bounds
        expected = (y2 - y1, x2 - x1)
        if hook_roi.shape[:2] != expected:
            raise ValueError(
                f"Hook ROI shape {hook_roi.shape[:2]} does not match {expected}"
            )
        self._canvas[y1:y2, x1:x2] = hook_roi
        return self._canvas

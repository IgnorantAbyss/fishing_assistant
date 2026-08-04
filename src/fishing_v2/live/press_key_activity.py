"""Read-only W/A/S/D transition telemetry for PRESS diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from src.fishing_v2.live.windows_action_sink import (
    CtypesWindowsInputApi,
    VIRTUAL_KEYS,
)


class KeyStateReader(Protocol):
    def is_down(self, key: str) -> bool: ...


class WindowsAsyncKeyStateReader:
    """Polling-only adapter; it installs no hook and emits no input."""

    def __init__(self, api: Any | None = None) -> None:
        self.api = api or CtypesWindowsInputApi()

    def is_down(self, key: str) -> bool:
        return bool(self.api.panic_pressed(VIRTUAL_KEYS[str(key).upper()]))


@dataclass(frozen=True)
class PressKeyActivity:
    frame_index: int
    timestamp: float
    key: str
    runtime_state: str
    press_episode_id: int | None
    action_sink_press_emission_active: bool
    source: str
    previous_poll_timestamp: float | None = None
    current_poll_timestamp: float | None = None
    runtime_press_emission_overlap: bool = False
    runtime_press_emission: dict[str, Any] | None = None
    latest_runtime_press_emission: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "key": self.key,
            "runtime_state": self.runtime_state,
            "press_episode_id": self.press_episode_id,
            "action_sink_press_emission_active": (
                self.action_sink_press_emission_active
            ),
            "source": self.source,
            "previous_poll_timestamp": self.previous_poll_timestamp,
            "current_poll_timestamp": self.current_poll_timestamp,
            "runtime_press_emission_overlap": (
                self.runtime_press_emission_overlap
            ),
            "runtime_press_emission": self.runtime_press_emission,
            "latest_runtime_press_emission": (
                self.latest_runtime_press_emission
            ),
            "telemetry_only": True,
        }


@dataclass(frozen=True)
class RuntimePressEmission:
    action_id: str
    press_episode_id: int
    sequence: tuple[str, ...]
    emission_started_at: float
    emission_completed_at: float

    def payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "press_episode_id": self.press_episode_id,
            "sequence": list(self.sequence),
            "emission_started_at": self.emission_started_at,
            "emission_completed_at": self.emission_completed_at,
        }


class PressKeyActivityMonitor:
    """Report rising edges without classifying physical versus synthetic input."""

    keys = ("W", "A", "S", "D")

    def __init__(self, reader: KeyStateReader) -> None:
        self.reader = reader
        self._previous = {key: False for key in self.keys}
        self._previous_poll_timestamp: float | None = None
        self._runtime_emissions: deque[RuntimePressEmission] = deque(
            maxlen=32
        )

    def record_runtime_emission(
        self,
        *,
        action_id: str,
        press_episode_id: int,
        sequence: tuple[str, ...],
        emission_started_at: float,
        emission_completed_at: float,
    ) -> None:
        """Remember a completed synchronous emission for the next poll.

        This is diagnostic correlation only.  GetAsyncKeyState cannot provide
        the key-down timestamp or prove whether an observed transition was
        physical or synthetic.
        """
        started = float(emission_started_at)
        completed = float(emission_completed_at)
        if completed < started:
            raise ValueError("PRESS emission completion precedes its start")
        self._runtime_emissions.append(RuntimePressEmission(
            str(action_id),
            int(press_episode_id),
            tuple(str(key).upper() for key in sequence),
            started,
            completed,
        ))

    def _overlapping_emissions(
        self, current_poll_timestamp: float
    ) -> tuple[RuntimePressEmission, ...]:
        previous = self._previous_poll_timestamp
        if previous is None:
            return ()
        return tuple(
            emission
            for emission in self._runtime_emissions
            if (
                emission.emission_started_at <= current_poll_timestamp
                and emission.emission_completed_at > previous
            )
        )

    def poll(
        self,
        *,
        frame_index: int,
        timestamp: float,
        runtime_state: str,
        press_episode_id: int | None,
        action_sink_press_emission_active: bool,
    ) -> tuple[PressKeyActivity, ...]:
        current_poll_timestamp = float(timestamp)
        previous_poll_timestamp = self._previous_poll_timestamp
        overlapping = self._overlapping_emissions(current_poll_timestamp)
        latest = (
            self._runtime_emissions[-1]
            if self._runtime_emissions else None
        )
        activities: list[PressKeyActivity] = []
        for key in self.keys:
            down = bool(self.reader.is_down(key))
            if down and not self._previous[key]:
                matching = tuple(
                    emission
                    for emission in overlapping
                    if key in emission.sequence
                )
                correlated = matching[-1] if matching else None
                if action_sink_press_emission_active or correlated is not None:
                    source = "runtime_emission_correlated"
                elif overlapping:
                    source = "runtime_emission_or_external_ambiguous"
                else:
                    source = "external_or_manual_candidate"
                activities.append(PressKeyActivity(
                    int(frame_index),
                    current_poll_timestamp,
                    key,
                    str(runtime_state),
                    press_episode_id,
                    bool(action_sink_press_emission_active),
                    source,
                    previous_poll_timestamp,
                    current_poll_timestamp,
                    bool(overlapping),
                    (
                        correlated.payload()
                        if correlated is not None else
                        overlapping[-1].payload()
                        if overlapping else None
                    ),
                    latest.payload() if latest is not None else None,
                ))
            self._previous[key] = down
        self._previous_poll_timestamp = current_poll_timestamp
        while (
            len(self._runtime_emissions) > 1
            and self._runtime_emissions[0].emission_completed_at
            <= current_poll_timestamp
        ):
            self._runtime_emissions.popleft()
        return tuple(activities)

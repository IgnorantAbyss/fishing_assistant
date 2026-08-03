"""Read-only W/A/S/D transition telemetry for PRESS diagnostics."""

from __future__ import annotations

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
            "telemetry_only": True,
        }


class PressKeyActivityMonitor:
    """Report rising edges without classifying physical versus synthetic input."""

    keys = ("W", "A", "S", "D")

    def __init__(self, reader: KeyStateReader) -> None:
        self.reader = reader
        self._previous = {key: False for key in self.keys}

    def poll(
        self,
        *,
        frame_index: int,
        timestamp: float,
        runtime_state: str,
        press_episode_id: int | None,
        action_sink_press_emission_active: bool,
    ) -> tuple[PressKeyActivity, ...]:
        activities: list[PressKeyActivity] = []
        for key in self.keys:
            down = bool(self.reader.is_down(key))
            if down and not self._previous[key]:
                activities.append(PressKeyActivity(
                    int(frame_index),
                    float(timestamp),
                    key,
                    str(runtime_state),
                    press_episode_id,
                    bool(action_sink_press_emission_active),
                    (
                        "runtime_emission_correlated"
                        if action_sink_press_emission_active else
                        "external_or_manual_candidate"
                    ),
                ))
            self._previous[key] = down
        return tuple(activities)

"""Bounded non-blocking console events for latency-sensitive Live paths."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(frozen=True)
class ConsoleEvent:
    message: str
    heartbeat: bool = False


class LiveConsoleEventQueue:
    """Move console writes off detector and ActionSink call paths."""

    def __init__(
        self,
        *,
        max_events: int = 64,
        writer: Callable[[str], None] | None = None,
    ) -> None:
        if max_events < 1:
            raise ValueError("console max_events must be positive")
        self.max_events = int(max_events)
        self.writer = writer or (lambda value: print(value, flush=True))
        self._events: deque[ConsoleEvent] = deque()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._closing = False
        self._dropped_heartbeats = 0
        self._dropped_events = 0
        self._last_heartbeat_at: dict[str, float] = {}

    def _ensure_worker(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="fishing-live-console",
            daemon=True,
        )
        self._thread.start()

    def emit(self, message: str, *, heartbeat: bool = False) -> bool:
        event = ConsoleEvent(str(message), bool(heartbeat))
        with self._condition:
            if self._closing:
                return False
            self._ensure_worker()
            if len(self._events) >= self.max_events:
                if heartbeat:
                    self._dropped_heartbeats += 1
                    return False
                heartbeat_index = next(
                    (
                        index
                        for index, queued in enumerate(self._events)
                        if queued.heartbeat
                    ),
                    None,
                )
                if heartbeat_index is not None:
                    del self._events[heartbeat_index]
                    self._dropped_heartbeats += 1
                else:
                    self._events.popleft()
                    self._dropped_events += 1
            self._events.append(event)
            self._condition.notify()
        return True

    def heartbeat(
        self,
        key: str,
        message: str,
        *,
        timestamp: float,
        minimum_interval_seconds: float = 0.5,
    ) -> bool:
        previous = self._last_heartbeat_at.get(key)
        if (
            previous is not None
            and float(timestamp) - previous < minimum_interval_seconds
        ):
            return False
        self._last_heartbeat_at[key] = float(timestamp)
        return self.emit(message, heartbeat=True)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._events and not self._closing:
                    self._condition.wait()
                if not self._events and self._closing:
                    return
                event = self._events.popleft()
            try:
                self.writer(event.message)
            except Exception:
                # Console output is diagnostic only and can never stop Live.
                pass

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout_seconds))

    def summary(self) -> dict[str, int]:
        return {
            "console_dropped_heartbeats": self._dropped_heartbeats,
            "console_dropped_events": self._dropped_events,
            "console_pending_events": len(self._events),
        }

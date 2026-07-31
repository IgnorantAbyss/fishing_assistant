from __future__ import annotations

import threading

from src.fishing_v2.live.console_events import LiveConsoleEventQueue


def test_full_console_queue_drops_heartbeat_without_blocking_producer() -> None:
    release = threading.Event()
    writer_started = threading.Event()
    written: list[str] = []

    def slow_writer(message: str) -> None:
        writer_started.set()
        release.wait(timeout=1.0)
        written.append(message)

    queue = LiveConsoleEventQueue(max_events=2, writer=slow_writer)
    assert queue.emit("worker-blocker") is True
    assert writer_started.wait(timeout=1.0)
    assert queue.emit("heartbeat-1", heartbeat=True) is True
    assert queue.emit("heartbeat-2", heartbeat=True) is True
    assert queue.emit("heartbeat-dropped", heartbeat=True) is False
    # A critical action event evicts a heartbeat; it never waits for output.
    assert queue.emit("HOOK_ACTION applied") is True
    assert queue.summary()["console_dropped_heartbeats"] == 2
    release.set()
    queue.close()
    assert "HOOK_ACTION applied" in written


def test_hook_heartbeat_is_rate_limited_and_uses_only_console_writer() -> None:
    written: list[str] = []
    queue = LiveConsoleEventQueue(writer=written.append)
    assert queue.heartbeat(
        "hook",
        "HOOK critical active fps=40.0",
        timestamp=1.0,
        minimum_interval_seconds=0.5,
    ) is True
    assert queue.heartbeat(
        "hook",
        "HOOK critical active fps=41.0",
        timestamp=1.2,
        minimum_interval_seconds=0.5,
    ) is False
    assert queue.heartbeat(
        "hook",
        "HOOK critical active fps=39.5",
        timestamp=1.5,
        minimum_interval_seconds=0.5,
    ) is True
    queue.close()
    assert written == [
        "HOOK critical active fps=40.0",
        "HOOK critical active fps=39.5",
    ]

"""Capture-only screen access for explicitly invoked collection commands."""

from __future__ import annotations

from typing import Any

import numpy as np


def _mss_module() -> Any:
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("mss is required for screen capture; install requirements.txt") from exc
    return mss


def get_monitors() -> list[dict[str, int]]:
    """Return physical monitor bounds using the same one-based indexes as mss."""
    mss = _mss_module()
    try:
        with mss.mss() as capture:
            return [{"index": index, **dict(monitor)} for index, monitor in enumerate(capture.monitors[1:], start=1)]
    except Exception as exc:
        raise RuntimeError(f"Unable to enumerate monitors with mss: {exc}") from exc


def capture_screen(monitor_index: int = 1) -> np.ndarray:
    """Capture one physical monitor as a BGR numpy array without sending input."""
    if monitor_index < 1:
        raise ValueError("monitor_index must be a physical one-based monitor index (>= 1)")
    mss = _mss_module()
    try:
        with mss.mss() as capture:
            if monitor_index >= len(capture.monitors):
                available = len(capture.monitors) - 1
                raise ValueError(f"Monitor {monitor_index} does not exist; {available} physical monitor(s) available")
            bgra = np.asarray(capture.grab(capture.monitors[monitor_index]))
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Unable to capture monitor {monitor_index} with mss: {exc}") from exc
    # mss returns BGRA; discard alpha while retaining OpenCV-compatible BGR order.
    return np.ascontiguousarray(bgra[:, :, :3])

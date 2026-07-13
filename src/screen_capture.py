"""Capture-only screen access for explicitly invoked collection commands."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int
    window_handle: int | None = None
    window_title: str | None = None

    def as_mss_monitor(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


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


def find_window_region(window_title: str) -> CaptureRegion:
    """Resolve an exact visible Windows title without activating or focusing it."""
    if not window_title.strip():
        raise ValueError("window_title must be non-empty")
    if os.name != "nt":
        raise RuntimeError("Window-title capture is supported only on Windows")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    handle = int(user32.FindWindowW(None, window_title))
    if not handle or not user32.IsWindowVisible(handle):
        raise RuntimeError(f"Visible game window was not found by exact title: {window_title!r}")
    rectangle = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rectangle)):
        raise RuntimeError(f"Could not read game window bounds: {window_title!r}")
    width = int(rectangle.right - rectangle.left)
    height = int(rectangle.bottom - rectangle.top)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Game window has invalid bounds: {width}x{height}")
    return CaptureRegion(
        int(rectangle.left), int(rectangle.top), width, height, handle, window_title
    )


class MSSCaptureSession:
    """Persistent read-only MSS capture for one monitor or exact window region."""

    def __init__(self, *, window_title: str | None = None, monitor_index: int = 1) -> None:
        if not window_title and monitor_index < 1:
            raise ValueError("monitor_index must be >= 1")
        self.window_title = window_title
        self.monitor_index = monitor_index
        self.region: CaptureRegion | None = None
        self._capture: Any | None = None

    def open(self) -> CaptureRegion:
        if self._capture is not None:
            raise RuntimeError("Capture session is already open")
        mss = _mss_module()
        try:
            capture = mss.mss()
            if self.window_title:
                region = find_window_region(self.window_title)
            else:
                if self.monitor_index >= len(capture.monitors):
                    raise ValueError(
                        f"Monitor {self.monitor_index} does not exist; "
                        f"{len(capture.monitors) - 1} physical monitor(s) available"
                    )
                monitor = capture.monitors[self.monitor_index]
                region = CaptureRegion(
                    int(monitor["left"]), int(monitor["top"]),
                    int(monitor["width"]), int(monitor["height"]),
                )
            self._capture = capture
            self.region = region
            return region
        except Exception:
            if "capture" in locals():
                capture.close()
            raise

    def capture(self) -> np.ndarray:
        if self._capture is None or self.region is None:
            raise RuntimeError("Capture session is not open")
        try:
            bgra = np.asarray(self._capture.grab(self.region.as_mss_monitor()))
        except Exception as exc:
            raise RuntimeError(f"Live capture failed: {exc}") from exc
        return np.ascontiguousarray(bgra[:, :, :3])

    def is_foreground(self) -> bool | None:
        if self.region is None or self.region.window_handle is None:
            return None
        if os.name != "nt":
            return None
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        return int(user32.GetForegroundWindow()) == self.region.window_handle

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
        self._capture = None

    def __enter__(self) -> "MSSCaptureSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

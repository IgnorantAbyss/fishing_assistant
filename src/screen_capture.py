"""Capture-only screen access for explicitly invoked collection commands."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

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


@dataclass(frozen=True)
class WindowInfo:
    """Validated identity and client bounds for one exact Windows HWND."""

    window_handle: int
    window_title: str
    process_id: int
    process_name: str
    client_region: CaptureRegion
    visible: bool = True
    minimized: bool = False


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


def mss_bgra_to_bgr(frame: Any) -> np.ndarray:
    """Convert MSS BGRA to the canonical uint8 HxWx3 contiguous BGR contract."""
    bgra = np.asarray(frame)
    if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"MSS frame must be uint8 HxWx4 BGRA, got {bgra.dtype} {bgra.shape}")
    return np.ascontiguousarray(bgra[:, :, :3])


def surface_to_bgr(frame: Any, channel_order: str) -> np.ndarray:
    """Convert a capture surface once at the adapter boundary."""
    raw = np.asarray(frame)
    order = channel_order.upper()
    expected_channels = 4 if order in {"BGRA", "RGBA"} else 3
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != expected_channels:
        raise ValueError(
            f"Capture surface must be uint8 HxWx{expected_channels} {order}, "
            f"got {raw.dtype} {raw.shape}"
        )
    if order == "BGRA":
        converted = raw[:, :, :3]
    elif order == "RGBA":
        converted = raw[:, :, [2, 1, 0]]
    elif order == "BGR":
        converted = raw
    elif order == "RGB":
        converted = raw[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported capture channel order: {channel_order!r}")
    return validate_bgr_frame(np.ascontiguousarray(converted))


def validate_bgr_frame(frame: Any) -> np.ndarray:
    """Enforce the single live-capture contract consumed by perception."""
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise TypeError(f"Capture frame dtype must be uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Capture frame must be HxWx3 BGR, got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"Capture frame is empty: {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError("Capture frame must be C-contiguous")
    return array


def frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(memoryview(validate_bgr_frame(frame))).hexdigest()


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
            bgra = capture.grab(capture.monitors[monitor_index])
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Unable to capture monitor {monitor_index} with mss: {exc}") from exc
    # mss returns BGRA; discard alpha while retaining OpenCV-compatible BGR order.
    return mss_bgra_to_bgr(bgra)


def _find_window_handle_exact(window_title: str) -> int:
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
    user32.FindWindowW.restype = ctypes.c_void_p
    return int(user32.FindWindowW(None, window_title) or 0)


def _process_name(process_id: int) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    process = kernel32.OpenProcess(0x1000, False, process_id)
    if not process:
        raise RuntimeError(f"Could not query process {process_id}")
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            raise RuntimeError(f"Could not read executable path for process {process_id}")
        return Path(buffer.value).stem
    finally:
        kernel32.CloseHandle(process)


def inspect_window_handle(
    window_handle: int,
    *,
    expected_title: str | None = None,
    expected_process_name: str | None = None,
) -> WindowInfo:
    """Validate one existing HWND and read its client rectangle in screen coordinates."""
    if os.name != "nt":
        raise RuntimeError("Window-title capture is supported only on Windows")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    hwnd = wintypes.HWND(window_handle)
    user32.IsWindow.argtypes = (wintypes.HWND,)
    user32.IsWindow.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    if not user32.IsWindow(hwnd):
        raise RuntimeError(f"Window HWND is no longer valid: {window_handle}")
    title_length = int(user32.GetWindowTextLengthW(hwnd))
    title_buffer = ctypes.create_unicode_buffer(title_length + 1)
    user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
    title = title_buffer.value
    if expected_title is not None and title != expected_title:
        raise RuntimeError(
            f"Window title changed for HWND {window_handle}: expected {expected_title!r}, got {title!r}"
        )
    visible = bool(user32.IsWindowVisible(hwnd))
    minimized = bool(user32.IsIconic(hwnd))
    if not visible:
        raise RuntimeError(f"Window is not visible: {title!r}")
    if minimized:
        raise RuntimeError(f"Window is minimized: {title!r}")

    client = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client)):
        raise RuntimeError(f"Could not read client bounds: {title!r}")
    width = int(client.right - client.left)
    height = int(client.bottom - client.top)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Window client area has invalid bounds: {width}x{height}")
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise RuntimeError(f"Could not map client bounds to the desktop: {title!r}")

    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    process_name = _process_name(int(process_id.value))
    if expected_process_name and process_name.casefold() != expected_process_name.casefold():
        raise RuntimeError(
            f"Unexpected process for {title!r}: expected {expected_process_name}, got {process_name}"
        )
    region = CaptureRegion(
        int(origin.x), int(origin.y), width, height, int(window_handle), title
    )
    return WindowInfo(
        int(window_handle), title, int(process_id.value), process_name, region, visible, minimized
    )


def resolve_exact_window(
    window_title: str,
    *,
    expected_process_name: str | None = None,
    handle_lookup: Callable[[str], int] | None = None,
    inspector: Callable[..., WindowInfo] | None = None,
) -> WindowInfo:
    """Resolve only the complete title; no partial-title or first-window fallback."""
    if not window_title.strip():
        raise ValueError("window_title must be non-empty")
    if os.name != "nt":
        # Tests can inject platform-neutral lookup/inspection doubles.
        if handle_lookup is None or inspector is None:
            raise RuntimeError("Window-title capture is supported only on Windows")
    lookup = handle_lookup or _find_window_handle_exact
    inspect = inspector or inspect_window_handle
    handle = int(lookup(window_title))
    if not handle:
        raise RuntimeError(f"Visible game window was not found by exact title: {window_title!r}")
    return inspect(
        handle,
        expected_title=window_title,
        expected_process_name=expected_process_name,
    )


def find_window_region(window_title: str) -> CaptureRegion:
    """Compatibility helper returning the exact window's client-area desktop region."""
    return resolve_exact_window(window_title).client_region


class MSSCaptureSession:
    """Persistent desktop-region capture; this is not HWND surface capture."""

    backend_name = "mss-region"

    def __init__(
        self,
        *,
        window_title: str | None = None,
        monitor_index: int = 1,
        expected_process_name: str | None = None,
        window_lookup: Callable[..., WindowInfo] = resolve_exact_window,
        window_inspector: Callable[..., WindowInfo] = inspect_window_handle,
    ) -> None:
        if not window_title and monitor_index < 1:
            raise ValueError("monitor_index must be >= 1")
        self.window_title = window_title
        self.monitor_index = monitor_index
        self.expected_process_name = expected_process_name
        self._window_lookup = window_lookup
        self._window_inspector = window_inspector
        self.region: CaptureRegion | None = None
        self.window_info: WindowInfo | None = None
        self._capture: Any | None = None
        self._diagnostics: dict[str, Any] = {"backend": self.backend_name}

    def open(self) -> CaptureRegion:
        if self._capture is not None:
            raise RuntimeError("Capture session is already open")
        mss = _mss_module()
        try:
            capture = mss.mss()
            if self.window_title:
                info = self._window_lookup(
                    self.window_title, expected_process_name=self.expected_process_name
                )
                region = info.client_region
                self.window_info = info
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
            self._diagnostics.update({
                "hwnd": info.window_handle if self.window_info else None,
                "process": info.process_name if self.window_info else None,
                "process_id": info.process_id if self.window_info else None,
                "window_title": info.window_title if self.window_info else None,
                "capture_region": {
                    "left": region.left, "top": region.top,
                    "width": region.width, "height": region.height,
                },
                "client_size": [region.width, region.height] if self.window_info else None,
                "channel_order": "BGR",
                "desktop_region_capture": True,
                "overlay_capture_warning": bool(self.window_info),
                "fallback_used": False,
            })
            return region
        except Exception:
            if "capture" in locals():
                capture.close()
            raise

    def capture(self) -> np.ndarray:
        if self._capture is None or self.region is None:
            raise RuntimeError("Capture session is not open")
        if self.window_info is not None:
            current = self._window_inspector(
                self.window_info.window_handle,
                expected_title=self.window_info.window_title,
                expected_process_name=self.expected_process_name,
            )
            original_size = (self.window_info.client_region.width, self.window_info.client_region.height)
            current_size = (current.client_region.width, current.client_region.height)
            if current_size != original_size:
                raise RuntimeError(
                    f"Window client size changed from {original_size[0]}x{original_size[1]} "
                    f"to {current_size[0]}x{current_size[1]}"
                )
            self.window_info = current
            self.region = current.client_region
        try:
            bgra = self._capture.grab(self.region.as_mss_monitor())
        except Exception as exc:
            raise RuntimeError(f"Live capture failed: {exc}") from exc
        raw = np.asarray(bgra)
        frame = mss_bgra_to_bgr(raw)
        validate_bgr_frame(frame)
        self._diagnostics.update({
            "raw_capture_shape": list(raw.shape),
            "converted_shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "channel_order": "BGR",
            "frame_sha256": frame_sha256(frame),
            "capture_region": {
                "left": self.region.left, "top": self.region.top,
                "width": self.region.width, "height": self.region.height,
            },
        })
        return frame

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)

    def is_foreground(self) -> bool | None:
        if self.region is None or self.region.window_handle is None:
            return None
        if os.name != "nt":
            return None
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
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

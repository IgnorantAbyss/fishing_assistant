"""Explicit, read-only capture backends for the live detect-only command."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
from typing import Any, Callable, Protocol

import numpy as np

from src.screen_capture import (
    MSSCaptureSession,
    WindowInfo,
    frame_sha256,
    inspect_window_handle,
    query_foreground_window,
    resolve_exact_window,
    surface_to_bgr,
    validate_bgr_frame,
)


MSS_REGION_BACKEND = "mss-region"
WINDOWS_GRAPHICS_CAPTURE_BACKEND = "windows-graphics-capture"
CAPTURE_BACKENDS = (MSS_REGION_BACKEND, WINDOWS_GRAPHICS_CAPTURE_BACKEND)
EXPECTED_GAME_PROCESS = "BlackDesert64"
EXPECTED_CLIENT_SIZE = (2560, 1440)


class CaptureBackendError(RuntimeError):
    pass


class WindowsGraphicsCaptureUnavailable(CaptureBackendError):
    pass


@dataclass(frozen=True)
class CapturedSurface:
    pixels: np.ndarray
    channel_order: str = "BGRA"


class HWNDCaptureProvider(Protocol):
    """Native WGC bridge contract; implementations must bind the supplied HWND."""

    capture_item_hwnd: int

    def open(self, client_size: tuple[int, int]) -> None: ...

    def capture(self) -> CapturedSurface: ...

    def is_valid(self) -> bool: ...

    def close(self) -> None: ...


def _unavailable_wgc_provider(_window_handle: int) -> HWNDCaptureProvider:
    discovered = [
        name for name in ("windows_capture", "winrt", "winsdk")
        if importlib.util.find_spec(name) is not None
    ]
    detail = ", ".join(discovered) if discovered else "none"
    raise WindowsGraphicsCaptureUnavailable(
        "No verified HWND Windows Graphics Capture bridge is available for "
        f"Python {sys.version_info.major}.{sys.version_info.minor}; discovered bindings: {detail}. "
        "Use --capture-backend mss-region, or explicitly add --allow-mss-fallback."
    )


class WindowsGraphicsCaptureSession:
    """Adapter for an HWND-bound WGC provider; never substitutes desktop pixels."""

    backend_name = WINDOWS_GRAPHICS_CAPTURE_BACKEND

    def __init__(
        self,
        *,
        window_title: str,
        expected_process_name: str = EXPECTED_GAME_PROCESS,
        expected_client_size: tuple[int, int] = EXPECTED_CLIENT_SIZE,
        resolved_window: WindowInfo | None = None,
        expected_title_prefix: str | None = None,
        window_resolution_mode: str = "exact_title",
        provider_factory: Callable[[int], HWNDCaptureProvider] = _unavailable_wgc_provider,
        window_lookup: Callable[..., WindowInfo] = resolve_exact_window,
        window_inspector: Callable[..., WindowInfo] = inspect_window_handle,
    ) -> None:
        self.window_title = window_title
        self.expected_process_name = expected_process_name
        self.expected_client_size = expected_client_size
        self._resolved_window = resolved_window
        self.expected_title_prefix = expected_title_prefix
        self.window_resolution_mode = window_resolution_mode
        self._provider_factory = provider_factory
        self._window_lookup = window_lookup
        self._window_inspector = window_inspector
        self.window_info: WindowInfo | None = None
        self._provider: HWNDCaptureProvider | None = None
        self._diagnostics: dict[str, Any] = {
            "backend": self.backend_name,
            "fallback_used": False,
            "desktop_region_capture": False,
            "overlay_capture_warning": False,
        }
        self._foreground_unavailable_count = 0
        self._foreground_unavailable_active = False

    def open(self) -> WindowInfo:
        if self._provider is not None:
            raise CaptureBackendError("Capture session is already open")
        if self._resolved_window is not None:
            info = self._window_inspector(
                self._resolved_window.window_handle,
                expected_title=(
                    None
                    if self.window_resolution_mode == "process_name"
                    else self._resolved_window.window_title
                ),
                expected_title_prefix=self.expected_title_prefix,
                require_non_empty_title=(
                    self.window_resolution_mode == "process_name"
                ),
                expected_process_name=self.expected_process_name,
                expected_process_id=self._resolved_window.process_id,
            )
        else:
            info = self._window_lookup(
                self.window_title,
                expected_process_name=self.expected_process_name,
            )
        client_size = (info.client_region.width, info.client_region.height)
        if client_size != self.expected_client_size:
            raise CaptureBackendError(
                f"Unsupported window client size {client_size[0]}x{client_size[1]}; "
                f"expected {self.expected_client_size[0]}x{self.expected_client_size[1]}"
            )
        provider: HWNDCaptureProvider | None = None
        try:
            # The only identifier crossing the native adapter boundary is the
            # exact, already validated HWND. A provider must create its capture
            # item from this handle rather than resolving a title again.
            provider = self._provider_factory(info.window_handle)
            if int(provider.capture_item_hwnd) != info.window_handle:
                raise CaptureBackendError(
                    "Windows Graphics Capture provider bound a different HWND: "
                    f"expected {info.window_handle}, got {provider.capture_item_hwnd}"
                )
            provider.open(client_size)
        except Exception:
            if provider is not None:
                provider.close()
            raise
        self.window_info = info
        self._provider = provider
        self._diagnostics.update({
            "hwnd": info.window_handle,
            "capture_item_hwnd": provider.capture_item_hwnd,
            "process": info.process_name,
            "process_id": info.process_id,
            "window_title": info.window_title,
            "window_title_prefix": self.expected_title_prefix,
            "window_resolution_mode": (
                self.window_resolution_mode
            ),
            "client_size": list(client_size),
            "capture_region": "hwnd-client-content",
            "channel_order": "BGR",
        })
        return info

    def capture(self) -> np.ndarray:
        if self._provider is None or self.window_info is None:
            raise CaptureBackendError("Capture session is not open")
        current = self._window_inspector(
            self.window_info.window_handle,
            expected_title=(
                None
                if self.window_resolution_mode == "process_name"
                else self.window_info.window_title
            ),
            expected_title_prefix=self.expected_title_prefix,
            require_non_empty_title=(
                self.window_resolution_mode == "process_name"
            ),
            expected_process_name=self.expected_process_name,
            expected_process_id=self.window_info.process_id,
        )
        current_size = (current.client_region.width, current.client_region.height)
        if current_size != self.expected_client_size:
            raise CaptureBackendError(
                f"Window client size changed to {current_size[0]}x{current_size[1]}"
            )
        if not self._provider.is_valid():
            raise CaptureBackendError("WGC capture item is invalid")
        surface = self._provider.capture()
        raw = np.asarray(surface.pixels)
        frame = surface_to_bgr(raw, surface.channel_order)
        height, width = frame.shape[:2]
        if (width, height) != self.expected_client_size:
            raise CaptureBackendError(
                f"Windows Graphics Capture returned {width}x{height}; "
                f"expected client content {self.expected_client_size[0]}x{self.expected_client_size[1]}"
            )
        self.window_info = current
        self._diagnostics.update({
            "raw_capture_shape": list(raw.shape),
            "raw_channel_order": surface.channel_order.upper(),
            "converted_shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "channel_order": "BGR",
            "frame_sha256": frame_sha256(frame),
        })
        return validate_bgr_frame(frame)

    def is_foreground(self) -> bool:
        if self.window_info is None:
            return False
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.argtypes = ()
        user32.GetForegroundWindow.restype = wintypes.HWND
        matches, unavailable, foreground_hwnd = query_foreground_window(
            self.window_info.window_handle, user32.GetForegroundWindow
        )
        if unavailable and not self._foreground_unavailable_active:
            self._foreground_unavailable_count += 1
        self._foreground_unavailable_active = unavailable
        self._diagnostics.update({
            "foreground": matches,
            "foreground_hwnd": foreground_hwnd,
            "foreground_window_unavailable": unavailable,
            "foreground_unavailable_count": self._foreground_unavailable_count,
        })
        return matches

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()
        self._provider = None


class ExplicitFallbackCaptureSession:
    """One-time initialization fallback; active capture failures always stop safely."""

    def __init__(self, primary: Any, fallback: Any, *, allowed: bool) -> None:
        self.primary = primary
        self.fallback = fallback
        self.allowed = allowed
        self.active: Any | None = None
        self.fallback_reason: str | None = None

    @property
    def backend_name(self) -> str:
        return self.active.backend_name if self.active is not None else self.primary.backend_name

    def open(self) -> Any:
        try:
            result = self.primary.open()
            self.active = self.primary
            return result
        except Exception as exc:
            self.primary.close()
            if not self.allowed:
                raise
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            result = self.fallback.open()
            self.active = self.fallback
            return result

    def capture(self) -> np.ndarray:
        if self.active is None:
            raise CaptureBackendError("Capture session is not open")
        return self.active.capture()

    def is_foreground(self) -> bool | None:
        if self.active is None:
            return False
        return self.active.is_foreground()

    def diagnostics(self) -> dict[str, Any]:
        base = self.active.diagnostics() if self.active is not None else {}
        return {
            **base,
            "requested_backend": self.primary.backend_name,
            "fallback_used": self.active is self.fallback,
            "fallback_reason": self.fallback_reason,
        }

    def close(self) -> None:
        if self.active is not None:
            self.active.close()
        self.active = None


def create_live_capture_session(
    *,
    backend: str,
    window_title: str,
    allow_mss_fallback: bool = False,
    expected_process_name: str = EXPECTED_GAME_PROCESS,
    resolved_window: WindowInfo | None = None,
    expected_title_prefix: str | None = None,
    window_resolution_mode: str = "exact_title",
    wgc_provider_factory: Callable[[int], HWNDCaptureProvider] = _unavailable_wgc_provider,
    window_lookup: Callable[..., WindowInfo] = resolve_exact_window,
    window_inspector: Callable[..., WindowInfo] = inspect_window_handle,
) -> Any:
    """Build an explicit backend; monitor capture is never an implicit option."""
    if backend not in CAPTURE_BACKENDS:
        raise ValueError(f"Unsupported capture backend: {backend!r}")
    common = {
        "window_title": window_title,
        "expected_process_name": expected_process_name,
        "resolved_window": resolved_window,
        "expected_title_prefix": expected_title_prefix,
        "window_resolution_mode": window_resolution_mode,
        "window_lookup": window_lookup,
        "window_inspector": window_inspector,
    }
    mss_session = MSSCaptureSession(**common)
    if backend == MSS_REGION_BACKEND:
        return mss_session
    wgc_session = WindowsGraphicsCaptureSession(
        **common, provider_factory=wgc_provider_factory
    )
    return ExplicitFallbackCaptureSession(
        wgc_session, mss_session, allowed=allow_mss_fallback
    )

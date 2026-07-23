"""Fail-closed resolution of one Windows top-level game window."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from typing import Callable, Iterable

from src.screen_capture import (
    CaptureRegion,
    WindowInfo,
    _process_name,
    normalize_process_name,
    resolve_exact_window,
)


WindowEnumerator = Callable[[], Iterable[WindowInfo]]


@dataclass(frozen=True)
class ResolvedWindowTarget:
    window_info: WindowInfo
    resolution_mode: str
    requested_process_name: str
    title_prefix: str | None = None

    @property
    def window_handle(self) -> int:
        return self.window_info.window_handle

    @property
    def process_id(self) -> int:
        return self.window_info.process_id

    @property
    def process_name(self) -> str:
        return self.window_info.process_name

    @property
    def window_title(self) -> str:
        return self.window_info.window_title

    @property
    def client_size(self) -> tuple[int, int]:
        region = self.window_info.client_region
        return region.width, region.height


class WindowResolutionError(RuntimeError):
    def __init__(
        self,
        reason: str,
        message: str,
        candidates: Iterable[WindowInfo] = (),
    ) -> None:
        self.reason = reason
        self.candidates = tuple(candidates)
        super().__init__(message)


def format_window_candidates(candidates: Iterable[WindowInfo]) -> str:
    rows = tuple(candidates)
    if not rows:
        return "  (none)"
    return "\n".join(
        "  "
        f"PID={item.process_id} "
        f"HWND={item.window_handle} "
        f"title={item.window_title!r} "
        f"client={item.client_region.width}x{item.client_region.height}"
        for item in rows
    )


def _read_top_level_window(window_handle: int) -> WindowInfo:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    hwnd = wintypes.HWND(window_handle)
    user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = (
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    )
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.RECT),
    )
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.POINT),
    )
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    )
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    title_length = int(user32.GetWindowTextLengthW(hwnd))
    title_buffer = ctypes.create_unicode_buffer(title_length + 1)
    user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
    title = title_buffer.value
    visible = bool(user32.IsWindowVisible(hwnd))
    minimized = bool(user32.IsIconic(hwnd))

    client = wintypes.RECT()
    width = height = 0
    if user32.GetClientRect(hwnd, ctypes.byref(client)):
        width = int(client.right - client.left)
        height = int(client.bottom - client.top)
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        origin = wintypes.POINT(0, 0)

    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    pid = int(process_id.value)
    process_name = _process_name(pid) if pid > 0 else ""
    region = CaptureRegion(
        int(origin.x),
        int(origin.y),
        width,
        height,
        int(window_handle),
        title,
    )
    return WindowInfo(
        int(window_handle),
        title,
        pid,
        process_name,
        region,
        visible,
        minimized,
    )


def enumerate_top_level_windows(
    *,
    reader: Callable[[int], WindowInfo] = _read_top_level_window,
) -> tuple[WindowInfo, ...]:
    if os.name != "nt":
        raise RuntimeError("Automatic window resolution is supported only on Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    windows: list[WindowInfo] = []

    @callback_type
    def callback(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
        try:
            raw_handle = getattr(hwnd, "value", hwnd)
            windows.append(reader(int(raw_handle or 0)))
        except Exception:
            # One inaccessible desktop window must not abort enumeration.
            pass
        return True

    user32.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
    user32.EnumWindows.restype = wintypes.BOOL
    if not user32.EnumWindows(callback, 0):
        raise RuntimeError("EnumWindows failed")
    return tuple(windows)


def resolve_window_target(
    *,
    exact_title: str | None = None,
    process_name: str | None = None,
    title_prefix: str | None = None,
    expected_process_name: str | None = None,
    enumerator: WindowEnumerator = enumerate_top_level_windows,
    exact_resolver: Callable[..., WindowInfo] = resolve_exact_window,
) -> ResolvedWindowTarget:
    """Resolve exactly one target and retain its immutable HWND/PID identity."""
    exact = exact_title.strip() if exact_title else None
    requested_process = process_name.strip() if process_name else None
    prefix = title_prefix if title_prefix else None
    if bool(exact) == bool(requested_process):
        raise ValueError(
            "Specify exactly one of exact_title or process_name"
        )
    if exact is not None:
        if prefix is not None:
            raise ValueError("title_prefix is only valid with process_name")
        info = exact_resolver(
            exact,
            expected_process_name=expected_process_name,
        )
        return ResolvedWindowTarget(
            info,
            "exact_title",
            expected_process_name or info.process_name,
        )

    assert requested_process is not None
    normalized_process = normalize_process_name(requested_process)
    candidates = tuple(
        item
        for item in enumerator()
        if item.visible
        and not item.minimized
        and bool(item.window_title.strip())
        and item.client_region.width > 0
        and item.client_region.height > 0
        and normalize_process_name(item.process_name) == normalized_process
        and (prefix is None or item.window_title.startswith(prefix))
    )
    if len(candidates) != 1:
        reason = (
            "window_candidate_not_found"
            if not candidates
            else "window_candidate_ambiguous"
        )
        raise WindowResolutionError(
            reason,
            (
                f"Automatic window resolution found {len(candidates)} candidates "
                f"for process {requested_process!r}"
                + (f" and title prefix {prefix!r}" if prefix else "")
            ),
            candidates,
        )
    return ResolvedWindowTarget(
        candidates[0],
        "process_name",
        requested_process,
        prefix,
    )

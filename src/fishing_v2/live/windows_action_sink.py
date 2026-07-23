"""Guarded foreground-only Windows SendInput action sink.

The implementation intentionally has no window-activation, background-message,
mouse, injection, or anti-cheat bypass path. The native adapter is constructed
only after the CLI explicitly enables the ``sendinput`` sink.
"""

from __future__ import annotations

from collections import Counter, deque
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import struct
import time
from typing import Any, Callable, Mapping, Protocol

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.ports.action_sink import (
    ActionExecutionContext,
    ActionExecutionResult,
)
from src.screen_capture import normalize_process_name


ACTION_SINK_NONE = "none"
ACTION_SINK_SENDINPUT = "sendinput"
ACTION_SINKS = (ACTION_SINK_NONE, ACTION_SINK_SENDINPUT)
EXPECTED_GAME_PROCESS = "BlackDesert64"
EXPECTED_CLIENT_SIZE = (2560, 1440)
INPUT_MODES = ("vk", "scancode")


class ActionIntegrityPreflightError(RuntimeError):
    def __init__(
        self, reason: str, message: str, diagnostics: Mapping[str, Any]
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.diagnostics = dict(diagnostics)


VIRTUAL_KEYS = {
    "SPACE": 0x20,
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "R": 0x52,
    "F12": 0x7B,
}


@dataclass(frozen=True)
class WindowsActionConfig:
    input_mode: str = "vk"
    key_hold_ms: int = 40
    sequence_interval_ms: int = 60
    minimum_action_interval_ms: int = 150
    panic_key: str = "F12"
    max_actions_per_minute: int = 30

    def __post_init__(self) -> None:
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        if self.key_hold_ms < 1:
            raise ValueError("key_hold_ms must be positive")
        if self.sequence_interval_ms < 0:
            raise ValueError("sequence_interval_ms must be non-negative")
        if self.minimum_action_interval_ms < 0:
            raise ValueError("minimum_action_interval_ms must be non-negative")
        if self.max_actions_per_minute < 1:
            raise ValueError("max_actions_per_minute must be positive")
        if self.panic_key.upper() not in VIRTUAL_KEYS:
            raise ValueError(f"Unsupported panic key: {self.panic_key!r}")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        panic_key: str | None = None,
    ) -> "WindowsActionConfig":
        data = dict(values or {})
        if panic_key is not None:
            data["panic_key"] = panic_key
        return cls(**data)


@dataclass(frozen=True)
class WindowSafetySnapshot:
    target_hwnd: int
    exists: bool
    visible: bool
    minimized: bool
    title: str
    process_id: int
    process_name: str
    client_size: tuple[int, int]
    foreground_hwnd: int | None


class WindowsInputApi(Protocol):
    def inspect_window(self, hwnd: int) -> WindowSafetySnapshot: ...

    def send_key_event(
        self, virtual_key: int, *, key_up: bool, input_mode: str
    ) -> "SendInputCallResult": ...

    def panic_pressed(self, virtual_key: int) -> bool: ...

    def process_integrity_diagnostics(self, target_process_id: int) -> Mapping[str, Any]: ...


def process_architecture() -> str:
    return f"{platform.machine() or 'unknown'}/{struct.calcsize('P') * 8}-bit"


@dataclass(frozen=True)
class SendInputCallResult:
    return_count: int
    windows_error_code: int
    windows_error_message: str
    input_count: int
    cb_size: int
    input_struct_size: int
    process_architecture: str
    input_mode: str
    virtual_key: int
    scan_code: int
    flags: int


# Windows uses LLP64. WPARAM is the pointer-width unsigned integer required
# for ULONG_PTR even though DWORD/ULONG remain 32-bit on 64-bit Windows.
ULONG_PTR = wintypes.WPARAM


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    )


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


class CtypesWindowsInputApi:
    """Small ctypes boundary that uses only documented Win32 APIs."""

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    MAPVK_VK_TO_VSC = 0
    SENDINPUT_INPUT_COUNT = 1
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    TOKEN_QUERY = 0x0008
    TOKEN_ELEVATION = 20
    TOKEN_INTEGRITY_LEVEL = 25

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows SendInput is available only on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self.user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(_INPUT),
            ctypes.c_int,
        )
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.IsWindow.argtypes = (wintypes.HWND,)
        self.user32.IsWindow.restype = wintypes.BOOL
        self.user32.IsWindowVisible.argtypes = (wintypes.HWND,)
        self.user32.IsWindowVisible.restype = wintypes.BOOL
        self.user32.IsIconic.argtypes = (wintypes.HWND,)
        self.user32.IsIconic.restype = wintypes.BOOL
        self.user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = (
            wintypes.HWND, wintypes.LPWSTR, ctypes.c_int,
        )
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self.user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD),
        )
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.GetClientRect.argtypes = (
            wintypes.HWND, ctypes.POINTER(wintypes.RECT),
        )
        self.user32.GetClientRect.restype = wintypes.BOOL
        self.user32.GetForegroundWindow.argtypes = ()
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
        self.user32.MapVirtualKeyW.restype = wintypes.UINT
        self.kernel32.OpenProcess.argtypes = (
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
        )
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        self.kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
        )
        self.advapi32.OpenProcessToken.restype = wintypes.BOOL
        self.advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        )
        self.advapi32.GetTokenInformation.restype = wintypes.BOOL
        self.advapi32.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
        self.advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.advapi32.GetSidSubAuthority.argtypes = (ctypes.c_void_p, wintypes.DWORD)
        self.advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)

    @staticmethod
    def _format_windows_error(error_code: int) -> str:
        try:
            return ctypes.FormatError(error_code).strip()
        except Exception:
            return f"Windows error {error_code}"

    def _process_name(self, process_id: int) -> str:
        process = self.kernel32.OpenProcess(
            self.PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
        )
        if not process:
            raise RuntimeError(f"OpenProcess failed for PID {process_id}")
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buffer))
            if not self.kernel32.QueryFullProcessImageNameW(
                process, 0, buffer, ctypes.byref(size)
            ):
                raise RuntimeError(f"QueryFullProcessImageNameW failed for PID {process_id}")
            return Path(buffer.value).stem
        finally:
            self.kernel32.CloseHandle(process)

    def inspect_window(self, hwnd: int) -> WindowSafetySnapshot:
        handle = wintypes.HWND(hwnd)
        foreground_result = self.user32.GetForegroundWindow()
        foreground = int(foreground_result) if foreground_result else None
        exists = bool(self.user32.IsWindow(handle))
        if not exists:
            return WindowSafetySnapshot(
                hwnd, False, False, False, "", 0, "", (0, 0), foreground
            )
        visible = bool(self.user32.IsWindowVisible(handle))
        minimized = bool(self.user32.IsIconic(handle))
        title_length = int(self.user32.GetWindowTextLengthW(handle))
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        self.user32.GetWindowTextW(handle, title_buffer, len(title_buffer))
        process_id = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
        client = wintypes.RECT()
        if not self.user32.GetClientRect(handle, ctypes.byref(client)):
            raise RuntimeError("GetClientRect failed")
        size = (int(client.right - client.left), int(client.bottom - client.top))
        return WindowSafetySnapshot(
            hwnd,
            exists,
            visible,
            minimized,
            title_buffer.value,
            int(process_id.value),
            self._process_name(int(process_id.value)),
            size,
            foreground,
        )

    def send_key_event(
        self,
        virtual_key: int,
        *,
        key_up: bool,
        input_mode: str,
    ) -> SendInputCallResult:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        scan_code = int(self.user32.MapVirtualKeyW(virtual_key, self.MAPVK_VK_TO_VSC))
        flags = self.KEYEVENTF_KEYUP if key_up else 0
        if input_mode == "scancode":
            flags |= self.KEYEVENTF_SCANCODE
        event = _INPUT(
            type=self.INPUT_KEYBOARD,
            ki=_KEYBDINPUT(
                wVk=virtual_key if input_mode == "vk" else 0,
                wScan=scan_code if input_mode == "scancode" else 0,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            ),
        )
        cb_size = ctypes.sizeof(_INPUT)
        ctypes.set_last_error(0)
        return_count = int(self.user32.SendInput(
            self.SENDINPUT_INPUT_COUNT,
            ctypes.pointer(event),
            cb_size,
        ))
        # Capture immediately. No other Win32 call is permitted before this.
        error_code = int(ctypes.get_last_error())
        error_message = self._format_windows_error(error_code)
        return SendInputCallResult(
            return_count=return_count,
            windows_error_code=error_code,
            windows_error_message=error_message,
            input_count=self.SENDINPUT_INPUT_COUNT,
            cb_size=cb_size,
            input_struct_size=ctypes.sizeof(_INPUT),
            process_architecture=process_architecture(),
            input_mode=input_mode,
            virtual_key=virtual_key,
            scan_code=scan_code,
            flags=flags,
        )

    def panic_pressed(self, virtual_key: int) -> bool:
        # Polling only: no global keyboard hook is installed.
        return bool(int(self.user32.GetAsyncKeyState(virtual_key)) & 0x8001)

    @staticmethod
    def _integrity_name(rid: int) -> str:
        if rid < 0x1000:
            return "untrusted"
        if rid < 0x2000:
            return "low"
        if rid < 0x3000:
            return "medium_plus" if rid >= 0x2100 else "medium"
        if rid < 0x4000:
            return "high"
        if rid < 0x5000:
            return "system"
        return "protected"

    def _query_process_integrity(self, process_id: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "process_id": int(process_id),
            "integrity_level": "unknown",
            "integrity_rid": None,
            "elevated": None,
            "error": None,
        }
        process = self.kernel32.OpenProcess(
            self.PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
        )
        if not process:
            code = int(ctypes.get_last_error())
            result["error"] = {
                "windows_error_code": code,
                "windows_error_message": self._format_windows_error(code),
            }
            return result
        token = wintypes.HANDLE()
        try:
            if not self.advapi32.OpenProcessToken(
                process, self.TOKEN_QUERY, ctypes.byref(token)
            ):
                code = int(ctypes.get_last_error())
                result["error"] = {
                    "windows_error_code": code,
                    "windows_error_message": self._format_windows_error(code),
                }
                return result
            elevation = wintypes.DWORD()
            returned = wintypes.DWORD()
            if self.advapi32.GetTokenInformation(
                token, self.TOKEN_ELEVATION, ctypes.byref(elevation),
                ctypes.sizeof(elevation), ctypes.byref(returned),
            ):
                result["elevated"] = bool(elevation.value)
            required = wintypes.DWORD()
            self.advapi32.GetTokenInformation(
                token, self.TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(required)
            )
            if required.value <= 0:
                return result
            buffer = ctypes.create_string_buffer(required.value)
            if not self.advapi32.GetTokenInformation(
                token, self.TOKEN_INTEGRITY_LEVEL, buffer,
                required.value, ctypes.byref(required),
            ):
                return result
            # TOKEN_MANDATORY_LABEL begins with SID_AND_ATTRIBUTES. On both
            # architectures its first field is the SID pointer.
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p)).contents.value
            if not sid:
                return result
            count_pointer = self.advapi32.GetSidSubAuthorityCount(sid)
            if not count_pointer or count_pointer.contents.value < 1:
                return result
            index = int(count_pointer.contents.value) - 1
            rid_pointer = self.advapi32.GetSidSubAuthority(sid, index)
            if not rid_pointer:
                return result
            rid = int(rid_pointer.contents.value)
            result["integrity_rid"] = rid
            result["integrity_level"] = self._integrity_name(rid)
            return result
        finally:
            if token:
                self.kernel32.CloseHandle(token)
            self.kernel32.CloseHandle(process)

    def process_integrity_diagnostics(self, target_process_id: int) -> Mapping[str, Any]:
        python_process = self._query_process_integrity(os.getpid())
        target_process = self._query_process_integrity(target_process_id)
        python_rid = python_process.get("integrity_rid")
        target_rid = target_process.get("integrity_rid")
        suspected_mismatch = (
            python_rid < target_rid
            if isinstance(python_rid, int) and isinstance(target_rid, int)
            else None
        )
        return {
            "python_process": python_process,
            "target_process": target_process,
            "suspected_integrity_mismatch": suspected_mismatch,
        }


def parse_action_allowlist(value: str | tuple[str, ...] | list[str]) -> frozenset[ActionIntent]:
    parts = value.split(",") if isinstance(value, str) else list(value)
    result: set[ActionIntent] = set()
    for item in parts:
        normalized = str(item).strip().upper()
        if not normalized:
            continue
        try:
            intent = ActionIntent(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported action allowlist entry: {item!r}") from exc
        if intent == ActionIntent.NONE:
            raise ValueError("NONE cannot be action-allowlisted")
        result.add(intent)
    return frozenset(result)


class WindowsSendInputActionSink:
    sink_type = ACTION_SINK_SENDINPUT

    def __init__(
        self,
        *,
        target_hwnd: int,
        expected_title: str,
        expected_title_prefix: str | None = None,
        window_resolution_mode: str = "exact_title",
        expected_process_id: int,
        expected_process_name: str = EXPECTED_GAME_PROCESS,
        expected_client_size: tuple[int, int] = EXPECTED_CLIENT_SIZE,
        allowlist: frozenset[ActionIntent] | None = None,
        config: WindowsActionConfig | None = None,
        api: WindowsInputApi | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        session_started_at: float = 0.0,
    ) -> None:
        if target_hwnd <= 0:
            raise ValueError("target_hwnd must be a positive exact HWND")
        if not expected_title:
            raise ValueError("expected_title must be non-empty")
        if expected_process_id <= 0:
            raise ValueError("expected_process_id must be positive")
        if window_resolution_mode not in {"exact_title", "process_name"}:
            raise ValueError("unsupported window_resolution_mode")
        self.target_hwnd = int(target_hwnd)
        self.expected_title = expected_title
        self.expected_title_prefix = expected_title_prefix
        self.window_resolution_mode = window_resolution_mode
        self.expected_process_id = int(expected_process_id)
        self.expected_process_name = expected_process_name
        self.expected_client_size = expected_client_size
        self.allowlist = frozenset(allowlist or ())
        self.config = config or WindowsActionConfig()
        self.api = api or CtypesWindowsInputApi()
        self.clock = clock
        self.sleep = sleep
        self.event_callback = event_callback
        self.session_started_at = float(session_started_at)
        self._panic_triggered = False
        self._focus_suspended = False
        self._applied_action_ids: set[str] = set()
        self._nonretryable_action_ids: dict[str, str] = {}
        self._attempt_times: deque[float] = deque()
        self._last_attempt_at: float | None = None
        self._attempted: Counter[str] = Counter()
        self._applied: Counter[str] = Counter()
        self._rejected: Counter[str] = Counter()
        self._partial: Counter[str] = Counter()
        self._failed: Counter[str] = Counter()
        self._rejection_reasons: Counter[str] = Counter()
        self._focus_loss_count = 0
        self._foreground_unavailable_count = 0
        self._foreground_unavailable_active = False
        integrity_reader = getattr(self.api, "process_integrity_diagnostics", None)
        if callable(integrity_reader):
            try:
                self._integrity_diagnostics = dict(
                    integrity_reader(self.expected_process_id)
                )
            except Exception as exc:
                self._integrity_diagnostics = {
                    "python_process": {"process_id": os.getpid(), "integrity_level": "unknown"},
                    "target_process": {
                        "process_id": self.expected_process_id,
                        "integrity_level": "unknown",
                    },
                    "suspected_integrity_mismatch": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            self._integrity_diagnostics = {
                "python_process": {"process_id": os.getpid(), "integrity_level": "unknown"},
                "target_process": {
                    "process_id": self.expected_process_id,
                    "integrity_level": "unknown",
                },
                "suspected_integrity_mismatch": None,
            }
        self._validate_integrity_preflight()
        self._event("action_sink_initialized", {
            "timestamp": self._timestamp(),
            "action_sink_type": self.sink_type,
            "target_hwnd": self.target_hwnd,
            "expected_title": self.expected_title,
            "expected_title_prefix": self.expected_title_prefix,
            "window_resolution_mode": self.window_resolution_mode,
            "expected_process_id": self.expected_process_id,
            "expected_process": self.expected_process_name,
            "expected_client_size": list(self.expected_client_size),
            "action_allowlist": sorted(item.value for item in self.allowlist),
            "panic_key": self.config.panic_key.upper(),
            "input_mode": self.config.input_mode,
            "action_applied_semantics": "complete_os_input_not_visual_acknowledgement",
            "input_struct_size": ctypes.sizeof(_INPUT),
            "python_pointer_size": ctypes.sizeof(ctypes.c_void_p),
            "process_architecture": process_architecture(),
            "integrity_diagnostics": self._integrity_diagnostics,
        })

    def _validate_integrity_preflight(self) -> None:
        python_process = self._integrity_diagnostics.get("python_process", {})
        target_process = self._integrity_diagnostics.get("target_process", {})
        python_rid = python_process.get("integrity_rid")
        target_rid = target_process.get("integrity_rid")
        if not isinstance(python_rid, int) or not isinstance(target_rid, int):
            raise ActionIntegrityPreflightError(
                "integrity_unknown",
                "Process integrity could not be verified; real input is fail-closed.",
                self._integrity_diagnostics,
            )
        if python_rid < target_rid:
            raise ActionIntegrityPreflightError(
                "integrity_mismatch",
                "Python process integrity is lower than target process integrity. "
                "Start the runtime manually from an elevated PowerShell.",
                self._integrity_diagnostics,
            )

    def _timestamp(self) -> float:
        return max(0.0, float(self.clock()) - self.session_started_at)

    def _event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, payload)

    @staticmethod
    def _key_sequence(request: ActionRequest) -> tuple[str, ...]:
        if request.intent in {
            ActionIntent.CAST, ActionIntent.START_HOOK, ActionIntent.HOOK_ACTION,
        }:
            return ("SPACE",)
        if request.intent == ActionIntent.COLLECT:
            return ("R",)
        if request.intent == ActionIntent.PRESS_SEQUENCE:
            raw = request.payload.get("sequence", ())
            if isinstance(raw, str):
                sequence = tuple(raw.upper())
            else:
                sequence = tuple(str(item).upper() for item in raw)
            if not sequence or any(item not in {"W", "A", "S", "D"} for item in sequence):
                raise ValueError("invalid_press_sequence")
            return sequence
        raise ValueError("unsupported_action_intent")

    def poll_panic(self, context: ActionExecutionContext | None = None) -> bool:
        if self._panic_triggered:
            return True
        panic_vk = VIRTUAL_KEYS[self.config.panic_key.upper()]
        if not self.api.panic_pressed(panic_vk):
            return False
        self._panic_triggered = True
        self._event("panic_stop", {
            "timestamp": self._timestamp(),
            "action_id": context.action_id if context else None,
            "episode_id": context.episode_id if context else None,
            "capture_frame_index": context.capture_frame_index if context else None,
            "runtime_state": context.runtime_state if context else None,
            "panic_key": self.config.panic_key.upper(),
            "action_applied": False,
        })
        return True

    def _result(
        self,
        request: ActionRequest,
        context: ActionExecutionContext,
        *,
        started_at: float | None,
        emitted: int,
        expected: int,
        foreground_hwnd: int | None,
        success: bool,
        applied: bool,
        rejection_reason: str | None = None,
        error: str | None = None,
        partial: bool = False,
        calls: tuple[SendInputCallResult, ...] = (),
        virtual_key: int | None = None,
    ) -> ActionExecutionResult:
        last_call = calls[-1] if calls else None
        return ActionExecutionResult(
            action_id=context.action_id,
            intent_type=request.intent.value,
            requested_at=context.requested_at,
            started_at=started_at,
            completed_at=self._timestamp(),
            success=success,
            applied=applied,
            emitted_event_count=emitted,
            expected_event_count=expected,
            target_hwnd=self.target_hwnd,
            foreground_hwnd=foreground_hwnd,
            rejection_reason=rejection_reason,
            error=error,
            partial_execution=partial,
            os_input_emitted=applied,
            windows_error_code=(
                last_call.windows_error_code if last_call is not None else None
            ),
            windows_error_message=(
                last_call.windows_error_message if last_call is not None else None
            ),
            sendinput_input_count=(
                last_call.input_count if last_call is not None else 0
            ),
            sendinput_cb_size=(
                last_call.cb_size if last_call is not None else ctypes.sizeof(_INPUT)
            ),
            input_struct_size=(
                last_call.input_struct_size
                if last_call is not None else ctypes.sizeof(_INPUT)
            ),
            process_architecture=(
                last_call.process_architecture
                if last_call is not None else process_architecture()
            ),
            input_mode=self.config.input_mode,
            virtual_key=(
                last_call.virtual_key if last_call is not None else virtual_key
            ),
            scan_code=(last_call.scan_code if last_call is not None else None),
            input_flags=tuple(call.flags for call in calls),
            integrity_diagnostics=self._integrity_diagnostics,
        )

    def _base_payload(
        self,
        request: ActionRequest,
        context: ActionExecutionContext,
        keys: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "intent": request.intent.value,
            "action_id": context.action_id,
            "episode_id": context.episode_id,
            "capture_frame_index": context.capture_frame_index,
            "runtime_state": context.runtime_state,
            "target_hwnd": self.target_hwnd,
            "key_sequence": list(keys),
            "timestamp": context.requested_at,
            "input_mode": self.config.input_mode,
            "process_architecture": process_architecture(),
            "input_struct_size": ctypes.sizeof(_INPUT),
        }

    def _reject(
        self,
        request: ActionRequest,
        context: ActionExecutionContext,
        reason: str,
        *,
        keys: tuple[str, ...] = (),
        foreground_hwnd: int | None = None,
        error: str | None = None,
    ) -> ActionExecutionResult:
        self._rejected[request.intent.value] += 1
        self._rejection_reasons[reason] += 1
        result = self._result(
            request, context, started_at=None, emitted=0,
            expected=len(keys) * 2, foreground_hwnd=foreground_hwnd,
            success=False, applied=False, rejection_reason=reason, error=error,
            virtual_key=VIRTUAL_KEYS[keys[0]] if keys else None,
        )
        self._event("action_rejected", {
            **self._base_payload(request, context, keys),
            **asdict(result),
        })
        return result

    def _safety_rejection(
        self, snapshot: WindowSafetySnapshot, context: ActionExecutionContext
    ) -> str | None:
        if context.target_hwnd != self.target_hwnd:
            return "target_hwnd_mismatch"
        if not snapshot.exists:
            return "target_window_invalid"
        if not snapshot.visible:
            return "target_window_not_visible"
        if snapshot.minimized:
            return "target_window_minimized"
        if self.window_resolution_mode == "process_name":
            if not snapshot.title:
                return "window_title_empty"
            if (
                self.expected_title_prefix is not None
                and not snapshot.title.startswith(self.expected_title_prefix)
            ):
                return "window_title_prefix_mismatch"
        elif snapshot.title != self.expected_title:
            return "window_title_mismatch"
        if snapshot.process_id != self.expected_process_id:
            return "target_process_id_mismatch"
        if (
            normalize_process_name(snapshot.process_name)
            != normalize_process_name(self.expected_process_name)
        ):
            return "process_name_mismatch"
        if snapshot.client_size != self.expected_client_size:
            return "client_size_mismatch"
        if snapshot.foreground_hwnd is None:
            return "foreground_window_unavailable"
        if snapshot.foreground_hwnd != self.target_hwnd:
            return "foreground_window_mismatch"
        return None

    def _rate_limit_rejection(self, now: float) -> str | None:
        while self._attempt_times and now - self._attempt_times[0] >= 60.0:
            self._attempt_times.popleft()
        if (
            self._last_attempt_at is not None
            and (now - self._last_attempt_at) * 1000.0
            < self.config.minimum_action_interval_ms
        ):
            return "minimum_action_interval"
        if len(self._attempt_times) >= self.config.max_actions_per_minute:
            return "max_actions_per_minute"
        return None

    def apply(
        self,
        intent: ActionRequest,
        context: ActionExecutionContext,
    ) -> ActionExecutionResult:
        request = intent
        if self.poll_panic(context):
            return self._reject(request, context, "panic_triggered")
        if request.intent not in self.allowlist:
            return self._reject(request, context, "action_not_allowlisted")
        if context.action_id in self._applied_action_ids:
            return self._reject(request, context, "duplicate_action")
        if context.action_id in self._nonretryable_action_ids:
            return self._reject(
                request, context, self._nonretryable_action_ids[context.action_id]
            )
        try:
            keys = self._key_sequence(request)
        except ValueError as exc:
            return self._reject(request, context, str(exc))
        try:
            snapshot = self.api.inspect_window(self.target_hwnd)
        except Exception as exc:
            return self._reject(
                request, context, "window_validation_error", keys=keys,
                error=f"{type(exc).__name__}: {exc}",
            )
        rejection = self._safety_rejection(snapshot, context)
        if rejection == "foreground_window_unavailable":
            if not self._foreground_unavailable_active:
                self._foreground_unavailable_count += 1
                self._event("foreground_window_unavailable", {
                    **self._base_payload(request, context, keys),
                    "foreground_hwnd": None,
                    "foreground_unavailable_count": self._foreground_unavailable_count,
                    "action_applied": False,
                })
            self._foreground_unavailable_active = True
        else:
            self._foreground_unavailable_active = False
        if rejection == "foreground_window_mismatch":
            if not self._focus_suspended:
                self._focus_suspended = True
                self._focus_loss_count += 1
                self._event("focus_lost", {
                    **self._base_payload(request, context, keys),
                    "foreground_hwnd": snapshot.foreground_hwnd,
                    "action_applied": False,
                })
        elif rejection is None and self._focus_suspended:
            self._focus_suspended = False
            self._event("focus_restored", {
                **self._base_payload(request, context, keys),
                "foreground_hwnd": snapshot.foreground_hwnd,
                "action_applied": False,
            })
        if rejection is not None:
            return self._reject(
                request, context, rejection, keys=keys,
                foreground_hwnd=snapshot.foreground_hwnd,
            )
        now = self.clock()
        rate_rejection = self._rate_limit_rejection(now)
        if rate_rejection:
            return self._reject(
                request, context, rate_rejection, keys=keys,
                foreground_hwnd=snapshot.foreground_hwnd,
            )
        self._event("action_allowed", {
            **self._base_payload(request, context, keys),
            "foreground_hwnd": snapshot.foreground_hwnd,
            "action_applied": False,
        })
        attempt_clock = self.clock()
        started_at = self._timestamp()
        self._last_attempt_at = attempt_clock
        self._attempt_times.append(attempt_clock)
        self._attempted[request.intent.value] += 1
        expected = len(keys) * 2
        emitted = 0
        calls: list[SendInputCallResult] = []
        self._event("action_started", {
            **self._base_payload(request, context, keys),
            "foreground_hwnd": snapshot.foreground_hwnd,
            "expected_event_count": expected,
            "action_applied": False,
        })
        error: str | None = None
        try:
            for index, key in enumerate(keys):
                if self.poll_panic(context):
                    error = "panic_triggered_during_sequence"
                    break
                down = self.api.send_key_event(
                    VIRTUAL_KEYS[key], key_up=False,
                    input_mode=self.config.input_mode,
                )
                calls.append(down)
                emitted += max(0, down.return_count)
                if down.return_count != 1:
                    error = (
                        f"SendInput key-down returned {down.return_count}, expected 1; "
                        f"Windows error {down.windows_error_code}: "
                        f"{down.windows_error_message}"
                    )
                    break
                self.sleep(self.config.key_hold_ms / 1000.0)
                up = self.api.send_key_event(
                    VIRTUAL_KEYS[key], key_up=True,
                    input_mode=self.config.input_mode,
                )
                calls.append(up)
                emitted += max(0, up.return_count)
                if up.return_count != 1:
                    error = (
                        f"SendInput key-up returned {up.return_count}, expected 1; "
                        f"Windows error {up.windows_error_code}: "
                        f"{up.windows_error_message}"
                    )
                    break
                if index + 1 < len(keys):
                    if self.poll_panic(context):
                        error = "panic_triggered_during_sequence"
                        break
                    self.sleep(self.config.sequence_interval_ms / 1000.0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        applied = emitted == expected and error is None
        partial = 0 < emitted < expected
        result = self._result(
            request, context, started_at=started_at, emitted=emitted,
            expected=expected, foreground_hwnd=snapshot.foreground_hwnd,
            success=applied, applied=applied,
            rejection_reason=None if applied else "sendinput_incomplete",
            error=error, partial=partial, calls=tuple(calls),
            virtual_key=VIRTUAL_KEYS[keys[0]] if keys else None,
        )
        payload = {
            **self._base_payload(request, context, keys),
            **asdict(result),
            "sendinput_return_count": emitted,
            "opportunity_result": (
                "os_input_emitted" if applied
                else "partial_not_applied" if partial
                else "failed_not_applied"
            ),
        }
        if applied:
            self._applied_action_ids.add(context.action_id)
            self._applied[request.intent.value] += 1
            self._event("action_applied", payload)
        elif partial:
            self._nonretryable_action_ids[context.action_id] = "partial_action_not_retried"
            self._partial[request.intent.value] += 1
            self._event("action_partial", payload)
        else:
            self._nonretryable_action_ids[context.action_id] = "failed_action_not_retried"
            self._failed[request.intent.value] += 1
            self._event("action_failed", payload)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "action_sink_type": self.sink_type,
            "action_allowlist": sorted(item.value for item in self.allowlist),
            "attempted_action_counts": dict(self._attempted),
            "applied_action_counts": dict(self._applied),
            "os_input_emitted_counts": dict(self._applied),
            "rejected_action_counts": dict(self._rejected),
            "partial_action_counts": dict(self._partial),
            "failed_action_counts": dict(self._failed),
            "rejection_counts_by_reason": dict(self._rejection_reasons),
            "panic_triggered": self._panic_triggered,
            "focus_loss_count": self._focus_loss_count,
            "foreground_unavailable_count": self._foreground_unavailable_count,
            "input_mode": self.config.input_mode,
            "input_struct_size": ctypes.sizeof(_INPUT),
            "process_architecture": process_architecture(),
            "integrity_diagnostics": self._integrity_diagnostics,
        }

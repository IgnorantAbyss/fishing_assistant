from __future__ import annotations

import ctypes
from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.live.windows_action_sink import (
    ActionIntegrityPreflightError,
    EXPECTED_CLIENT_SIZE,
    EXPECTED_GAME_PROCESS,
    CtypesWindowsInputApi,
    ULONG_PTR,
    SendInputCallResult,
    VIRTUAL_KEYS,
    WindowSafetySnapshot,
    WindowsActionConfig,
    WindowsSendInputActionSink,
    _HARDWAREINPUT,
    _INPUT,
    _INPUTUNION,
    _KEYBDINPUT,
    _MOUSEINPUT,
    parse_action_allowlist,
    process_architecture,
)
from src.fishing_v2.ports.action_sink import ActionExecutionContext
from tools.run_live_detect_only import parse_args


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeWindowsApi:
    def __init__(self) -> None:
        self.snapshot = WindowSafetySnapshot(
            4242, True, True, False, "test-window", 99,
            EXPECTED_GAME_PROCESS, EXPECTED_CLIENT_SIZE, 4242,
        )
        self.snapshots: list[WindowSafetySnapshot] = []
        self.send_results: list[int] = []
        self.send_calls: list[tuple[int, bool]] = []
        self.input_modes: list[str] = []
        self.panic_values: list[bool] = []
        self.inspect_calls = 0
        self.integrity_diagnostics = {
            "python_process": {
                "process_id": 1, "integrity_level": "medium",
                "integrity_rid": 8192, "elevated": False,
            },
            "target_process": {
                "process_id": 99, "integrity_level": "medium",
                "integrity_rid": 8192, "elevated": False,
            },
            "suspected_integrity_mismatch": False,
        }

    def inspect_window(self, _hwnd: int) -> WindowSafetySnapshot:
        self.inspect_calls += 1
        return self.snapshots.pop(0) if self.snapshots else self.snapshot

    def send_key_event(
        self, virtual_key: int, *, key_up: bool, input_mode: str
    ) -> SendInputCallResult:
        self.send_calls.append((virtual_key, key_up))
        self.input_modes.append(input_mode)
        returned = self.send_results.pop(0) if self.send_results else 1
        scan_codes = {
            VIRTUAL_KEYS["R"]: 0x13, VIRTUAL_KEYS["SPACE"]: 0x39,
            VIRTUAL_KEYS["W"]: 0x11, VIRTUAL_KEYS["A"]: 0x1E,
            VIRTUAL_KEYS["S"]: 0x1F, VIRTUAL_KEYS["D"]: 0x20,
        }
        flags = (0x0008 if input_mode == "scancode" else 0) | (0x0002 if key_up else 0)
        return SendInputCallResult(
            returned, 87 if returned == 0 else 0,
            "The parameter is incorrect." if returned == 0 else "The operation completed successfully.",
            1, ctypes.sizeof(_INPUT), ctypes.sizeof(_INPUT),
            process_architecture(), input_mode,
            virtual_key, scan_codes.get(virtual_key, 0), flags,
        )

    def panic_pressed(self, _virtual_key: int) -> bool:
        return self.panic_values.pop(0) if self.panic_values else False

    def process_integrity_diagnostics(self, target_process_id: int):
        result = {
            **self.integrity_diagnostics,
            "python_process": dict(self.integrity_diagnostics["python_process"]),
            "target_process": {
                **self.integrity_diagnostics["target_process"],
                "process_id": target_process_id,
            },
        }
        return result


def _context(action_id: str = "cycle:1:COLLECT", *, intent: str = "GET") -> ActionExecutionContext:
    return ActionExecutionContext(action_id, "1", 1.25, 25, intent, 4242)


def _sink(
    api: FakeWindowsApi,
    clock: FakeClock | None = None,
    *,
    allowlist: str = "COLLECT",
    config: WindowsActionConfig | None = None,
    events: list[tuple[str, dict]] | None = None,
    expected_title_prefix: str | None = None,
    enable_live_press_sequence: bool | None = None,
) -> WindowsSendInputActionSink:
    clock = clock or FakeClock()
    event_rows = events if events is not None else []
    return WindowsSendInputActionSink(
        target_hwnd=4242,
        expected_title="test-window",
        expected_title_prefix=expected_title_prefix,
        window_resolution_mode=(
            "process_name"
            if expected_title_prefix is not None else "exact_title"
        ),
        expected_process_id=99,
        allowlist=parse_action_allowlist(allowlist),
        config=config or WindowsActionConfig(),
        api=api,
        clock=clock,
        sleep=clock.sleep,
        event_callback=lambda name, payload: event_rows.append((name, dict(payload))),
        enable_live_press_sequence=(
            ActionIntent.PRESS_SEQUENCE
            in parse_action_allowlist(allowlist)
            if enable_live_press_sequence is None
            else enable_live_press_sequence
        ),
    )


def _press_payload(sequence: tuple[str, ...], capacity: int) -> dict:
    return {
        "sequence": sequence,
        "slot_capacity": capacity,
        "active_press_episode": True,
        "panel_confirmed": True,
        "frozen_by_consensus": True,
    }


def test_cli_defaults_to_detect_only_with_no_sink() -> None:
    args = parse_args(["--window-title", "exact-title"])
    assert args.emit_actions is False
    assert args.action_sink == "none"
    assert args.action_allowlist == ""
    assert args.panic_key == "F12"
    assert args.enable_live_press_sequence is False


def test_explicit_collect_only_cli_contract() -> None:
    args = parse_args([
        "--window-title", "exact-title", "--emit-actions", "true",
        "--action-sink", "sendinput", "--action-allowlist", "COLLECT",
        "--panic-key", "F12",
    ])
    assert args.emit_actions is True
    assert args.action_sink == "sendinput"
    assert parse_action_allowlist(args.action_allowlist) == {ActionIntent.COLLECT}


def test_press_cli_opt_in_is_explicit() -> None:
    args = parse_args([
        "--window-title",
        "exact-title",
        "--emit-actions",
        "true",
        "--action-sink",
        "sendinput",
        "--action-allowlist",
        "PRESS_SEQUENCE",
        "--enable-live-press-sequence",
    ])
    assert args.enable_live_press_sequence is True


def test_allowlist_parser_is_comma_separated_and_strict() -> None:
    assert parse_action_allowlist("COLLECT, HOOK_ACTION") == {
        ActionIntent.COLLECT, ActionIntent.HOOK_ACTION,
    }
    with pytest.raises(ValueError, match="Unsupported"):
        parse_action_allowlist("COLLECT,CLICK")


def test_allowlist_parser_normalizes_cast_and_collect_case() -> None:
    assert parse_action_allowlist("cast, collect") == {
        ActionIntent.CAST, ActionIntent.COLLECT,
    }


def test_user32_is_loaded_with_last_error_enabled() -> None:
    source = inspect.getsource(CtypesWindowsInputApi.__init__)
    assert 'ctypes.WinDLL("user32", use_last_error=True)' in source
    assert "GetForegroundWindow.argtypes = ()" in source


@pytest.mark.parametrize("allowlist", ["START_HOOK", "HOOK_ACTION"])
def test_integrity_mismatch_fails_before_sink_becomes_sendable(
    allowlist: str,
) -> None:
    api = FakeWindowsApi()
    api.integrity_diagnostics["target_process"].update({
        "integrity_level": "high", "integrity_rid": 12288, "elevated": True,
    })
    events: list[tuple[str, dict]] = []
    with pytest.raises(ActionIntegrityPreflightError, match="elevated PowerShell") as exc:
        _sink(api, allowlist=allowlist, events=events)
    assert exc.value.reason == "integrity_mismatch"
    assert api.send_calls == []
    assert all(name != "action_sink_initialized" for name, _ in events)


@pytest.mark.parametrize(("python_rid", "target_rid"), [(12288, 12288), (8192, 8192)])
def test_equal_known_integrity_levels_pass_preflight(
    python_rid: int, target_rid: int
) -> None:
    api = FakeWindowsApi()
    api.integrity_diagnostics["python_process"]["integrity_rid"] = python_rid
    api.integrity_diagnostics["target_process"]["integrity_rid"] = target_rid
    sink = _sink(api)
    assert sink.summary()["integrity_diagnostics"]["python_process"][
        "integrity_rid"
    ] == python_rid


def test_unknown_integrity_fails_closed_for_real_sendinput() -> None:
    api = FakeWindowsApi()
    api.integrity_diagnostics["python_process"]["integrity_rid"] = None
    with pytest.raises(ActionIntegrityPreflightError) as exc:
        _sink(api)
    assert exc.value.reason == "integrity_unknown"
    assert api.send_calls == []


def test_windows_input_struct_layout_matches_pointer_architecture() -> None:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    assert ctypes.sizeof(ULONG_PTR) == pointer_size
    assert ctypes.sizeof(_HARDWAREINPUT) == 8
    if pointer_size == 8:
        assert ctypes.sizeof(_KEYBDINPUT) == 24
        assert ctypes.sizeof(_MOUSEINPUT) == 32
        assert ctypes.sizeof(_INPUTUNION) == 32
        assert ctypes.sizeof(_INPUT) == 40
    elif pointer_size == 4:
        assert ctypes.sizeof(_KEYBDINPUT) == 16
        assert ctypes.sizeof(_MOUSEINPUT) == 24
        assert ctypes.sizeof(_INPUTUNION) == 24
        assert ctypes.sizeof(_INPUT) == 28
    else:  # pragma: no cover - unsupported Windows architecture
        pytest.fail(f"Unsupported pointer size: {pointer_size}")


class CapturingUser32:
    def __init__(self, *, return_count: int, last_error: int = 0) -> None:
        self.return_count = return_count
        self.last_error = last_error
        self.calls = []

    @staticmethod
    def MapVirtualKeyW(virtual_key: int, _mode: int) -> int:
        return 0x13 if virtual_key == 0x52 else 0

    def SendInput(self, input_count, pointer, cb_size):
        event = pointer.contents
        self.calls.append({
            "input_count": input_count,
            "cb_size": cb_size,
            "type": event.type,
            "wVk": event.ki.wVk,
            "wScan": event.ki.wScan,
            "flags": event.ki.dwFlags,
            "extra": event.ki.dwExtraInfo,
        })
        ctypes.set_last_error(self.last_error)
        return self.return_count


def _ctypes_api_with(user32: CapturingUser32) -> CtypesWindowsInputApi:
    api = CtypesWindowsInputApi.__new__(CtypesWindowsInputApi)
    api.user32 = user32
    return api


def test_sendinput_zero_captures_last_error_and_correct_cb_size() -> None:
    user32 = CapturingUser32(return_count=0, last_error=87)
    result = _ctypes_api_with(user32).send_key_event(
        VIRTUAL_KEYS["R"], key_up=False, input_mode="vk"
    )
    assert result.return_count == 0
    assert result.windows_error_code == 87
    assert result.windows_error_message
    assert result.input_count == 1
    assert result.cb_size == result.input_struct_size == ctypes.sizeof(_INPUT)
    assert user32.calls[0]["cb_size"] == ctypes.sizeof(_INPUT)


def test_virtual_key_r_mapping_uses_vk_and_keyup_flags() -> None:
    down_user32 = CapturingUser32(return_count=1)
    down = _ctypes_api_with(down_user32).send_key_event(
        VIRTUAL_KEYS["R"], key_up=False, input_mode="vk"
    )
    up_user32 = CapturingUser32(return_count=1)
    up = _ctypes_api_with(up_user32).send_key_event(
        VIRTUAL_KEYS["R"], key_up=True, input_mode="vk"
    )
    assert down.virtual_key == up.virtual_key == 0x52
    assert down.scan_code == up.scan_code == 0x13
    assert down.flags == 0
    assert up.flags == CtypesWindowsInputApi.KEYEVENTF_KEYUP
    assert down_user32.calls[0]["wVk"] == 0x52
    assert down_user32.calls[0]["wScan"] == 0
    assert down_user32.calls[0]["extra"] == 0


def test_scancode_r_mapping_uses_scan_and_keyup_flags() -> None:
    down_user32 = CapturingUser32(return_count=1)
    down = _ctypes_api_with(down_user32).send_key_event(
        VIRTUAL_KEYS["R"], key_up=False, input_mode="scancode"
    )
    up_user32 = CapturingUser32(return_count=1)
    up = _ctypes_api_with(up_user32).send_key_event(
        VIRTUAL_KEYS["R"], key_up=True, input_mode="scancode"
    )
    assert down.flags == CtypesWindowsInputApi.KEYEVENTF_SCANCODE
    assert up.flags == (
        CtypesWindowsInputApi.KEYEVENTF_SCANCODE
        | CtypesWindowsInputApi.KEYEVENTF_KEYUP
    )
    assert down_user32.calls[0]["wVk"] == 0
    assert down_user32.calls[0]["wScan"] == 0x13


def test_scancode_sink_mode_does_not_duplicate_vk_events() -> None:
    api = FakeWindowsApi()
    sink = _sink(api, config=WindowsActionConfig(input_mode="scancode"))
    result = sink.apply(ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context())
    assert result.os_input_emitted is True
    assert api.input_modes == ["scancode", "scancode"]
    assert len(api.send_calls) == 2
    assert result.input_flags == (0x0008, 0x000A)


@pytest.mark.parametrize(
    "intent",
    [ActionIntent.CAST, ActionIntent.START_HOOK, ActionIntent.HOOK_ACTION],
)
def test_space_action_mappings_emit_one_down_and_up(intent: ActionIntent) -> None:
    api = FakeWindowsApi()
    sink = _sink(api, allowlist=intent.value)
    result = sink.apply(ActionRequest(intent, 0.99, "test"), _context(f"1:{intent.value}"))
    assert result.applied is True
    assert result.os_input_emitted is True
    assert api.send_calls == [(VIRTUAL_KEYS["SPACE"], False), (VIRTUAL_KEYS["SPACE"], True)]


def test_collect_mapping_emits_exactly_one_r_down_and_up() -> None:
    api = FakeWindowsApi()
    result = _sink(api).apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "qualified GET"), _context()
    )
    assert result.applied is True
    assert result.emitted_event_count == result.expected_event_count == 2
    assert api.send_calls == [(VIRTUAL_KEYS["R"], False), (VIRTUAL_KEYS["R"], True)]


def test_press_sequence_emits_only_frozen_wasd_in_order() -> None:
    api = FakeWindowsApi()
    request = ActionRequest(
        ActionIntent.PRESS_SEQUENCE,
        0.99,
        "frozen",
        _press_payload(("W", "A", "S", "D"), 8),
    )
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        request, _context("cycle:1:PRESS_SEQUENCE", intent="PRESS")
    )
    assert result.applied is True
    assert api.send_calls == [
        (VIRTUAL_KEYS[key], key_up)
        for key in ("W", "A", "S", "D")
        for key_up in (False, True)
    ]
    assert result.completed_key_count == result.total_key_count == 4
    assert result.attempted_count == 4


def test_press_sequence_preserves_repeated_letters() -> None:
    api = FakeWindowsApi()
    sequence = ("W", "W", "A", "D")
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        ActionRequest(
            ActionIntent.PRESS_SEQUENCE,
            0.99,
            "frozen",
            _press_payload(sequence, 8),
        ),
        _context("cycle:1:PRESS_SEQUENCE", intent="PRESS"),
    )
    assert result.applied is True
    assert api.send_calls == [
        (VIRTUAL_KEYS[key], key_up)
        for key in sequence
        for key_up in (False, True)
    ]


def test_press_sink_constructor_requires_explicit_opt_in() -> None:
    with pytest.raises(
        ValueError,
        match="--enable-live-press-sequence",
    ):
        _sink(
            FakeWindowsApi(),
            allowlist="PRESS_SEQUENCE",
            enable_live_press_sequence=False,
        )


def test_invalid_press_character_rejects_entire_sequence_before_input() -> None:
    api = FakeWindowsApi()
    request = ActionRequest(
        ActionIntent.PRESS_SEQUENCE,
        0.99,
        "bad",
        _press_payload(("W", "X", "D"), 8),
    )
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        request, _context("cycle:1:PRESS_SEQUENCE", intent="PRESS")
    )
    assert result.applied is False
    assert result.rejection_reason == "invalid_press_sequence"
    assert api.inspect_calls == 0
    assert api.send_calls == []


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_press_payload((), 8), "invalid_press_sequence"),
        (
            _press_payload(("W", "A", "S", "D"), 3),
            "press_sequence_exceeds_slot_capacity",
        ),
    ],
)
def test_invalid_press_shape_is_rejected_before_any_sendinput(
    payload: dict,
    reason: str,
) -> None:
    api = FakeWindowsApi()
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        ActionRequest(
            ActionIntent.PRESS_SEQUENCE,
            0.99,
            "invalid",
            payload,
        ),
        _context("cycle:invalid:PRESS_SEQUENCE", intent="PRESS"),
    )
    assert result.rejection_reason == reason
    assert api.send_calls == []


def test_action_not_allowlisted_is_rejected_without_window_or_input() -> None:
    api = FakeWindowsApi()
    result = _sink(api).apply(
        ActionRequest(ActionIntent.HOOK_ACTION, 0.99, "test"),
        _context("cycle:1:HOOK_ACTION", intent="HOOK"),
    )
    assert result.rejection_reason == "action_not_allowlisted"
    assert api.inspect_calls == 0
    assert api.send_calls == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"foreground_hwnd": 9999}, "foreground_window_mismatch"),
        ({"minimized": True}, "target_window_minimized"),
        ({"process_name": "notepad"}, "process_name_mismatch"),
        ({"process_id": 100}, "target_process_id_mismatch"),
        ({"client_size": (1280, 720)}, "client_size_mismatch"),
        ({"title": "different-title"}, "window_title_mismatch"),
        ({"visible": False}, "target_window_not_visible"),
        ({"exists": False}, "target_window_invalid"),
    ],
)
def test_window_safety_rejections_emit_no_input(change: dict, reason: str) -> None:
    api = FakeWindowsApi()
    api.snapshot = replace(api.snapshot, **change)
    result = _sink(api).apply(ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context())
    assert result.rejection_reason == reason
    assert result.applied is False
    assert api.send_calls == []


def test_auto_title_prefix_allows_suffix_change_but_fails_closed_on_prefix_loss() -> None:
    api = FakeWindowsApi()
    api.snapshot = replace(api.snapshot, title="黑色沙漠 - 999999")
    sink = _sink(api, expected_title_prefix="黑色沙漠")
    accepted = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"),
        _context("get:auto:accepted"),
    )
    assert accepted.applied is True
    assert len(api.send_calls) == 2

    api.send_calls.clear()
    api.snapshot = replace(api.snapshot, title="其他視窗 - 999999")
    rejected = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"),
        _context("get:auto:rejected"),
    )
    assert rejected.rejection_reason == "window_title_prefix_mismatch"
    assert rejected.applied is False
    assert api.send_calls == []


def test_null_foreground_rejects_without_input_and_throttles_event() -> None:
    api = FakeWindowsApi()
    api.snapshot = replace(api.snapshot, foreground_hwnd=None)
    events: list[tuple[str, dict]] = []
    sink = _sink(api, events=events)
    first = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context("get:1")
    )
    second = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context("get:2")
    )
    assert first.rejection_reason == second.rejection_reason == (
        "foreground_window_unavailable"
    )
    assert api.send_calls == []
    assert [name for name, _ in events].count("foreground_window_unavailable") == 1
    assert sink.summary()["foreground_unavailable_count"] == 1


def test_context_target_hwnd_must_match_startup_target() -> None:
    api = FakeWindowsApi()
    context = replace(_context(), target_hwnd=7777)
    result = _sink(api).apply(ActionRequest(ActionIntent.COLLECT, 0.99, "test"), context)
    assert result.rejection_reason == "target_hwnd_mismatch"
    assert api.send_calls == []


@pytest.mark.parametrize(
    ("intent", "state"),
    [
        (ActionIntent.START_HOOK, "READY"),
        (ActionIntent.HOOK_ACTION, "HOOK"),
    ],
)
def test_panic_key_permanently_disables_sink_for_session(
    intent: ActionIntent,
    state: str,
) -> None:
    api = FakeWindowsApi()
    api.panic_values = [True]
    events: list[tuple[str, dict]] = []
    sink = _sink(api, allowlist=intent.value, events=events)
    request = ActionRequest(intent, 0.99, "qualified")
    first = sink.apply(
        request,
        _context(f"cycle:1:{intent.value}", intent=state),
    )
    api.panic_values = [False]
    second = sink.apply(
        request,
        _context(f"cycle:2:{intent.value}", intent=state),
    )
    assert first.rejection_reason == second.rejection_reason == "panic_triggered"
    assert sum(name == "panic_stop" for name, _ in events) == 1
    assert api.send_calls == []


def test_panic_during_sequence_cancels_remaining_keys_as_partial() -> None:
    api = FakeWindowsApi()
    api.panic_values = [False, False, True]
    request = ActionRequest(
        ActionIntent.PRESS_SEQUENCE,
        0.99,
        "frozen",
        _press_payload(("W", "A"), 8),
    )
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        request, _context("cycle:1:PRESS_SEQUENCE", intent="PRESS")
    )
    assert result.applied is False
    assert result.partial_execution is True
    assert result.os_input_emitted is False
    assert result.emitted_event_count == 2
    assert result.completed_key_count == 1
    assert result.total_key_count == 2
    assert len(api.send_calls) == 2


def test_third_press_key_failure_stops_after_completed_prefix() -> None:
    api = FakeWindowsApi()
    api.send_results = [1, 1, 1, 1, 0]
    sink = _sink(api, allowlist="PRESS_SEQUENCE")
    request = ActionRequest(
        ActionIntent.PRESS_SEQUENCE,
        0.99,
        "frozen",
        _press_payload(("W", "A", "S", "D"), 8),
    )
    first = sink.apply(
        request,
        _context("cycle:third-fails", intent="PRESS"),
    )
    second = sink.apply(
        request,
        _context("cycle:third-fails", intent="PRESS"),
    )
    assert first.partial_execution is True
    assert first.attempted_count == 3
    assert first.completed_key_count == 2
    assert first.total_key_count == 4
    assert api.send_calls == [
        (VIRTUAL_KEYS["W"], False),
        (VIRTUAL_KEYS["W"], True),
        (VIRTUAL_KEYS["A"], False),
        (VIRTUAL_KEYS["A"], True),
        (VIRTUAL_KEYS["S"], False),
    ]
    assert second.rejection_reason == "partial_action_not_retried"


def test_press_focus_loss_before_third_key_stops_sequence() -> None:
    api = FakeWindowsApi()
    valid = api.snapshot
    api.snapshots = [
        valid,
        valid,
        valid,
        replace(valid, foreground_hwnd=9999),
    ]
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        ActionRequest(
            ActionIntent.PRESS_SEQUENCE,
            0.99,
            "frozen",
            _press_payload(("W", "A", "S", "D"), 8),
        ),
        _context("cycle:focus-loss", intent="PRESS"),
    )
    assert result.partial_execution is True
    assert result.completed_key_count == 2
    assert len(api.send_calls) == 4
    assert "foreground_window_mismatch" in str(result.error)


def test_press_diagnostics_are_flushed_after_all_key_events() -> None:
    order: list[str] = []

    class OrderedApi(FakeWindowsApi):
        def send_key_event(self, virtual_key, *, key_up, input_mode):
            order.append("send")
            return super().send_key_event(
                virtual_key,
                key_up=key_up,
                input_mode=input_mode,
            )

    api = OrderedApi()
    sink = _sink(
        api,
        allowlist="PRESS_SEQUENCE",
        events=[],
    )
    sink.event_callback = lambda name, _payload: order.append(
        f"event:{name}"
    )
    result = sink.apply(
        ActionRequest(
            ActionIntent.PRESS_SEQUENCE,
            0.99,
            "frozen",
            _press_payload(("W", "A"), 8),
        ),
        _context("cycle:deferred-diagnostics", intent="PRESS"),
    )
    assert result.applied is True
    last_send = max(
        index for index, item in enumerate(order) if item == "send"
    )
    first_action_event = min(
        index
        for index, item in enumerate(order)
        if item.startswith("event:action_")
    )
    assert first_action_event > last_send


def test_duplicate_applied_action_id_never_sends_twice() -> None:
    api = FakeWindowsApi()
    sink = _sink(api)
    request = ActionRequest(ActionIntent.COLLECT, 0.99, "test")
    first = sink.apply(request, _context())
    second = sink.apply(request, _context())
    assert first.applied is True
    assert second.applied is False
    assert second.rejection_reason == "duplicate_action"
    assert len(api.send_calls) == 2


def test_partial_sendinput_is_not_applied_and_is_not_retried() -> None:
    api = FakeWindowsApi()
    api.send_results = [1, 0]
    sink = _sink(api)
    request = ActionRequest(ActionIntent.COLLECT, 0.99, "test")
    first = sink.apply(request, _context())
    second = sink.apply(request, _context())
    assert first.partial_execution is True
    assert first.success is first.applied is False
    assert first.os_input_emitted is False
    assert second.rejection_reason == "partial_action_not_retried"
    assert len(api.send_calls) == 2


def test_zero_sendinput_return_is_failed_not_partial_or_applied() -> None:
    api = FakeWindowsApi()
    api.send_results = [0]
    events: list[tuple[str, dict]] = []
    result = _sink(api, events=events).apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context()
    )
    assert result.applied is False
    assert result.partial_execution is False
    assert result.emitted_event_count == 0
    assert result.os_input_emitted is False
    assert result.windows_error_code == 87
    assert result.windows_error_message == "The parameter is incorrect."
    assert any(name == "action_failed" for name, _ in events)


def test_zero_event_failure_is_not_retried_for_same_action_id() -> None:
    api = FakeWindowsApi()
    api.send_results = [0]
    sink = _sink(api)
    request = ActionRequest(ActionIntent.COLLECT, 0.99, "test")
    first = sink.apply(request, _context())
    second = sink.apply(request, _context())
    assert first.emitted_event_count == 0
    assert second.rejection_reason == "failed_action_not_retried"
    assert len(api.send_calls) == 1


def test_focus_loss_suspends_and_matching_foreground_restores() -> None:
    api = FakeWindowsApi()
    events: list[tuple[str, dict]] = []
    sink = _sink(api, events=events)
    api.snapshot = replace(api.snapshot, foreground_hwnd=9999)
    rejected = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context("cycle:1:COLLECT")
    )
    api.snapshot = replace(api.snapshot, foreground_hwnd=4242)
    restored = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context("cycle:2:COLLECT")
    )
    assert rejected.rejection_reason == "foreground_window_mismatch"
    assert restored.applied is True
    assert [name for name, _ in events].count("focus_lost") == 1
    assert [name for name, _ in events].count("focus_restored") == 1


def test_minimum_action_interval_limits_new_action_ids() -> None:
    api = FakeWindowsApi()
    clock = FakeClock()
    sink = _sink(api, clock)
    request = ActionRequest(ActionIntent.COLLECT, 0.99, "test")
    assert sink.apply(request, _context("cycle:1:COLLECT")).applied is True
    result = sink.apply(request, _context("cycle:2:COLLECT"))
    assert result.rejection_reason == "minimum_action_interval"
    assert len(api.send_calls) == 2


def test_max_actions_per_minute_limits_new_action_ids() -> None:
    api = FakeWindowsApi()
    clock = FakeClock()
    sink = _sink(api, clock, config=WindowsActionConfig(
        minimum_action_interval_ms=0, max_actions_per_minute=1,
    ))
    request = ActionRequest(ActionIntent.COLLECT, 0.99, "test")
    assert sink.apply(request, _context("cycle:1:COLLECT")).applied is True
    result = sink.apply(request, _context("cycle:2:COLLECT"))
    assert result.rejection_reason == "max_actions_per_minute"


def test_summary_contains_required_action_counters() -> None:
    api = FakeWindowsApi()
    sink = _sink(api)
    sink.apply(ActionRequest(ActionIntent.HOOK_ACTION, 0.9, "blocked"), _context("hook"))
    sink.apply(ActionRequest(ActionIntent.COLLECT, 0.9, "allowed"), _context())
    summary = sink.summary()
    assert summary["action_sink_type"] == "sendinput"
    assert summary["action_allowlist"] == ["COLLECT"]
    assert summary["attempted_action_counts"] == {"COLLECT": 1}
    assert summary["applied_action_counts"] == {"COLLECT": 1}
    assert summary["rejected_action_counts"] == {"HOOK_ACTION": 1}
    assert summary["partial_action_counts"] == {}
    assert summary["failed_action_counts"] == {}
    assert summary["rejection_counts_by_reason"] == {"action_not_allowlisted": 1}
    assert summary["panic_triggered"] is False
    assert summary["focus_loss_count"] == 0


def test_action_result_records_timestamps_hwnds_and_complete_counts() -> None:
    api = FakeWindowsApi()
    result = _sink(api).apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context()
    )
    assert result.action_id == "cycle:1:COLLECT"
    assert result.intent_type == "COLLECT"
    assert result.requested_at == 1.25
    assert result.started_at is not None
    assert result.completed_at >= result.started_at
    assert result.target_hwnd == result.foreground_hwnd == 4242
    assert result.success is result.applied is True
    assert result.os_input_emitted is True
    assert result.emitted_event_count == result.expected_event_count == 2
    assert result.rejection_reason is result.error is None
    assert result.partial_execution is False
    assert result.sendinput_input_count == 1
    assert result.sendinput_cb_size == result.input_struct_size == ctypes.sizeof(_INPUT)
    assert result.input_mode == "vk"
    assert result.virtual_key == 0x52
    assert result.scan_code == 0x13
    assert result.input_flags == (0, 0x0002)
    assert result.integrity_diagnostics["suspected_integrity_mismatch"] is False


def test_implementation_has_no_background_or_mouse_input_path() -> None:
    source = (Path(__file__).resolve().parents[2] / "src" / "fishing_v2" / "live" / "windows_action_sink.py").read_text(encoding="utf-8")
    normalized = source.casefold()
    assert "postmessage" not in normalized
    assert "setforegroundwindow" not in normalized
    assert "pyautogui" not in normalized
    assert "mouse_event" not in normalized

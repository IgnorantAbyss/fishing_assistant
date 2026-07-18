from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.live.windows_action_sink import (
    EXPECTED_CLIENT_SIZE,
    EXPECTED_GAME_PROCESS,
    VIRTUAL_KEYS,
    WindowSafetySnapshot,
    WindowsActionConfig,
    WindowsSendInputActionSink,
    parse_action_allowlist,
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
            4242, True, True, False, "黑色沙漠 - 525411", 99,
            EXPECTED_GAME_PROCESS, EXPECTED_CLIENT_SIZE, 4242,
        )
        self.send_results: list[int] = []
        self.send_calls: list[tuple[int, bool]] = []
        self.panic_values: list[bool] = []
        self.inspect_calls = 0

    def inspect_window(self, _hwnd: int) -> WindowSafetySnapshot:
        self.inspect_calls += 1
        return self.snapshot

    def send_key_event(self, virtual_key: int, *, key_up: bool) -> int:
        self.send_calls.append((virtual_key, key_up))
        return self.send_results.pop(0) if self.send_results else 1

    def panic_pressed(self, _virtual_key: int) -> bool:
        return self.panic_values.pop(0) if self.panic_values else False


def _context(action_id: str = "cycle:1:COLLECT", *, intent: str = "GET") -> ActionExecutionContext:
    return ActionExecutionContext(action_id, "1", 1.25, 25, intent, 4242)


def _sink(
    api: FakeWindowsApi,
    clock: FakeClock | None = None,
    *,
    allowlist: str = "COLLECT",
    config: WindowsActionConfig | None = None,
    events: list[tuple[str, dict]] | None = None,
) -> WindowsSendInputActionSink:
    clock = clock or FakeClock()
    event_rows = events if events is not None else []
    return WindowsSendInputActionSink(
        target_hwnd=4242,
        expected_title="黑色沙漠 - 525411",
        expected_process_id=99,
        allowlist=parse_action_allowlist(allowlist),
        config=config or WindowsActionConfig(),
        api=api,
        clock=clock,
        sleep=clock.sleep,
        event_callback=lambda name, payload: event_rows.append((name, dict(payload))),
    )


def test_cli_defaults_to_detect_only_with_no_sink() -> None:
    args = parse_args(["--window-title", "exact-title"])
    assert args.emit_actions is False
    assert args.action_sink == "none"
    assert args.action_allowlist == ""
    assert args.panic_key == "F12"


def test_explicit_collect_only_cli_contract() -> None:
    args = parse_args([
        "--window-title", "exact-title", "--emit-actions", "true",
        "--action-sink", "sendinput", "--action-allowlist", "COLLECT",
        "--panic-key", "F12",
    ])
    assert args.emit_actions is True
    assert args.action_sink == "sendinput"
    assert parse_action_allowlist(args.action_allowlist) == {ActionIntent.COLLECT}


def test_allowlist_parser_is_comma_separated_and_strict() -> None:
    assert parse_action_allowlist("COLLECT, HOOK_ACTION") == {
        ActionIntent.COLLECT, ActionIntent.HOOK_ACTION,
    }
    with pytest.raises(ValueError, match="Unsupported"):
        parse_action_allowlist("COLLECT,CLICK")


@pytest.mark.parametrize(
    "intent",
    [ActionIntent.CAST, ActionIntent.START_HOOK, ActionIntent.HOOK_ACTION],
)
def test_space_action_mappings_emit_one_down_and_up(intent: ActionIntent) -> None:
    api = FakeWindowsApi()
    sink = _sink(api, allowlist=intent.value)
    result = sink.apply(ActionRequest(intent, 0.99, "test"), _context(f"1:{intent.value}"))
    assert result.applied is True
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
        ActionIntent.PRESS_SEQUENCE, 0.99, "frozen", {"sequence": ("W", "A", "S", "D")}
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


def test_invalid_press_character_rejects_entire_sequence_before_input() -> None:
    api = FakeWindowsApi()
    request = ActionRequest(
        ActionIntent.PRESS_SEQUENCE, 0.99, "bad", {"sequence": ("W", "X", "D")}
    )
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        request, _context("cycle:1:PRESS_SEQUENCE", intent="PRESS")
    )
    assert result.applied is False
    assert result.rejection_reason == "invalid_press_sequence"
    assert api.inspect_calls == 0
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


def test_context_target_hwnd_must_match_startup_target() -> None:
    api = FakeWindowsApi()
    context = replace(_context(), target_hwnd=7777)
    result = _sink(api).apply(ActionRequest(ActionIntent.COLLECT, 0.99, "test"), context)
    assert result.rejection_reason == "target_hwnd_mismatch"
    assert api.send_calls == []


def test_panic_key_permanently_disables_sink_for_session() -> None:
    api = FakeWindowsApi()
    api.panic_values = [True]
    events: list[tuple[str, dict]] = []
    sink = _sink(api, events=events)
    first = sink.apply(ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context())
    api.panic_values = [False]
    second = sink.apply(
        ActionRequest(ActionIntent.COLLECT, 0.99, "test"), _context("cycle:2:COLLECT")
    )
    assert first.rejection_reason == second.rejection_reason == "panic_triggered"
    assert sum(name == "panic_stop" for name, _ in events) == 1
    assert api.send_calls == []


def test_panic_during_sequence_cancels_remaining_keys_as_partial() -> None:
    api = FakeWindowsApi()
    api.panic_values = [False, False, True]
    request = ActionRequest(
        ActionIntent.PRESS_SEQUENCE, 0.99, "frozen", {"sequence": "WA"}
    )
    result = _sink(api, allowlist="PRESS_SEQUENCE").apply(
        request, _context("cycle:1:PRESS_SEQUENCE", intent="PRESS")
    )
    assert result.applied is False
    assert result.partial_execution is True
    assert result.emitted_event_count == 2
    assert len(api.send_calls) == 2


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
    assert any(name == "action_failed" for name, _ in events)


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
    assert result.emitted_event_count == result.expected_event_count == 2
    assert result.rejection_reason is result.error is None
    assert result.partial_execution is False


def test_implementation_has_no_background_or_mouse_input_path() -> None:
    source = (Path(__file__).resolve().parents[2] / "src" / "fishing_v2" / "live" / "windows_action_sink.py").read_text(encoding="utf-8")
    normalized = source.casefold()
    assert "postmessage" not in normalized
    assert "setforegroundwindow" not in normalized
    assert "pyautogui" not in normalized
    assert "mouse_event" not in normalized

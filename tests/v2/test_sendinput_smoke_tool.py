from __future__ import annotations

import ast
import ctypes
import json
from pathlib import Path

import pytest

from src.fishing_v2.live.windows_action_sink import (
    SendInputCallResult,
    VIRTUAL_KEYS,
    WindowSafetySnapshot,
    _INPUT,
    process_architecture,
)
from src.screen_capture import CaptureRegion, WindowInfo
from tools.test_sendinput import parse_args, run


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(0.0, seconds)


class FakeApi:
    def __init__(self) -> None:
        self.snapshot = WindowSafetySnapshot(
            4242, True, True, False, "Untitled - Notepad", 99,
            "Notepad", (800, 600), 4242,
        )
        self.send_calls: list[tuple[int, bool, str]] = []
        self.panic_values: list[bool] = []

    def process_integrity_diagnostics(self, target_process_id: int):
        return {
            "python_process": {"process_id": 1, "integrity_level": "medium"},
            "target_process": {
                "process_id": target_process_id, "integrity_level": "medium",
            },
            "suspected_integrity_mismatch": False,
        }

    def panic_pressed(self, _virtual_key: int) -> bool:
        return self.panic_values.pop(0) if self.panic_values else False

    def inspect_window(self, _hwnd: int) -> WindowSafetySnapshot:
        return self.snapshot

    def send_key_event(self, virtual_key: int, *, key_up: bool, input_mode: str):
        self.send_calls.append((virtual_key, key_up, input_mode))
        flags = (0x0008 if input_mode == "scancode" else 0) | (0x0002 if key_up else 0)
        return SendInputCallResult(
            1, 0, "The operation completed successfully.", 1,
            ctypes.sizeof(_INPUT), ctypes.sizeof(_INPUT),
            process_architecture(), input_mode, virtual_key, 0x13, flags,
        )


def _window(*, process: str = "Notepad") -> WindowInfo:
    return WindowInfo(
        4242, "Untitled - Notepad", 99, process,
        CaptureRegion(0, 0, 800, 600, 4242, "Untitled - Notepad"),
    )


def _run(argv, api: FakeApi | None = None):
    api = api or FakeApi()
    clock = FakeClock()
    output: list[str] = []
    code = run(
        argv,
        api_factory=lambda: api,
        window_lookup=lambda _title: _window(),
        clock=clock,
        sleep=clock.sleep,
        output=output.append,
    )
    return code, json.loads(output[-1]), api


def test_smoke_tool_defaults_to_dry_run() -> None:
    args = parse_args([])
    assert args.dry_run is True
    assert args.key == "R"
    assert args.count == 1
    assert args.input_mode == "vk"


def test_dry_run_never_constructs_api_or_calls_sendinput() -> None:
    output = []
    code = run(
        [], api_factory=lambda: pytest.fail("API must not initialize in dry-run"),
        output=output.append,
    )
    payload = json.loads(output[-1])
    assert code == 0
    assert payload["result"] == "dry_run_no_input"
    assert payload["os_input_emitted"] is False


def test_non_dry_run_requires_exact_target_title() -> None:
    code, payload, api = _run(["--send", "--delay-seconds", "0"])
    assert code == 2
    assert payload["reason"] == "target_window_title_required"
    assert api.send_calls == []


def test_nonforeground_target_is_rejected_without_input() -> None:
    api = FakeApi()
    api.snapshot = WindowSafetySnapshot(
        4242, True, True, False, "Untitled - Notepad", 99,
        "Notepad", (800, 600), 7777,
    )
    code, payload, api = _run([
        "--send", "--target-window-title", "Untitled - Notepad",
        "--delay-seconds", "0",
    ], api)
    assert code == 2
    assert payload["reason"] == "foreground_window_mismatch"
    assert api.send_calls == []


def test_f12_stops_during_delay_without_input() -> None:
    api = FakeApi()
    api.panic_values = [True]
    code, payload, api = _run([
        "--send", "--target-window-title", "Untitled - Notepad",
        "--delay-seconds", "3",
    ], api)
    assert code == 3
    assert payload["result"] == "panic_stop"
    assert api.send_calls == []


@pytest.mark.parametrize("input_mode", ["vk", "scancode"])
def test_single_r_smoke_uses_selected_mode_once(input_mode: str) -> None:
    code, payload, api = _run([
        "--send", "--target-window-title", "Untitled - Notepad",
        "--key", "R", "--input-mode", input_mode,
        "--count", "1", "--delay-seconds", "0",
    ])
    assert code == 0
    assert payload["os_input_emitted"] is True
    assert payload["action_applied"] is False
    assert payload["visual_acknowledgement_required"] is True
    assert api.send_calls == [
        (VIRTUAL_KEYS["R"], False, input_mode),
        (VIRTUAL_KEYS["R"], True, input_mode),
    ]


def test_smoke_tool_refuses_game_process() -> None:
    output = []
    api = FakeApi()
    code = run(
        ["--send", "--target-window-title", "game", "--delay-seconds", "0"],
        api_factory=lambda: api,
        window_lookup=lambda _title: _window(process="BlackDesert64"),
        output=output.append,
    )
    payload = json.loads(output[-1])
    assert code == 2
    assert payload["reason"] == "smoke_tool_requires_non_game_test_window"
    assert api.send_calls == []


def test_smoke_tool_cannot_send_sequences_or_mouse() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--count", "2"])
    with pytest.raises(SystemExit):
        parse_args(["--key", "W"])
    path = Path(__file__).resolve().parents[2] / "tools" / "test_sendinput.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("detector" in item for item in imports)
    source = path.read_text(encoding="utf-8").casefold()
    assert "postmessage" not in source
    assert "setforegroundwindow" not in source
    assert "mouse_event" not in source

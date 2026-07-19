"""Safely validate one Windows SendInput key against a non-game test window.

This tool is dry-run by default, never starts Fishing Runtime, never imports a
detector, never changes foreground focus, and supports exactly one R key.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.live.windows_action_sink import (  # noqa: E402
    CtypesWindowsInputApi,
    EXPECTED_GAME_PROCESS,
    INPUT_MODES,
    VIRTUAL_KEYS,
    _INPUT,
    process_architecture,
)
from src.screen_capture import WindowInfo, resolve_exact_window  # noqa: E402


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Notepad flow: open Notepad, run with --send and its exact title, "
            "then click inside the editor during the delay and leave it foreground. "
            "Example: .venv\\Scripts\\python.exe tools\\test_sendinput.py "
            "--send --target-window-title \"<exact Notepad title>\" --key R "
            "--input-mode vk --count 1 --delay-seconds 3"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true")
    mode.add_argument(
        "--send", dest="dry_run", action="store_false",
        help="Explicitly permit one key only after foreground validation",
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument("--target-window-title")
    parser.add_argument("--key", choices=("R",), default="R")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="vk")
    parser.add_argument("--count", type=int, choices=(1,), default=1)
    parser.add_argument("--delay-seconds", type=_non_negative_float, default=3.0)
    return parser.parse_args(argv)


def _emit(output: Callable[[str], None], payload: dict[str, Any]) -> None:
    output(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[[], Any] = CtypesWindowsInputApi,
    window_lookup: Callable[[str], WindowInfo] = resolve_exact_window,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    output: Callable[[str], None] = print,
) -> int:
    args = parse_args(argv)
    base = {
        "dry_run": args.dry_run,
        "target_window_title": args.target_window_title,
        "key": args.key,
        "input_mode": args.input_mode,
        "count": args.count,
        "delay_seconds": args.delay_seconds,
        "virtual_key": VIRTUAL_KEYS[args.key],
        "input_struct_size": ctypes.sizeof(_INPUT),
        "process_architecture": process_architecture(),
        "os_input_emitted": False,
        "action_applied": False,
    }
    if args.dry_run:
        _emit(output, {**base, "result": "dry_run_no_input"})
        return 0
    if not args.target_window_title:
        _emit(output, {**base, "result": "refused", "reason": "target_window_title_required"})
        return 2
    try:
        target = window_lookup(args.target_window_title)
    except Exception as exc:
        _emit(output, {
            **base,
            "result": "refused",
            "reason": "target_window_resolution_failed",
            "error": f"{type(exc).__name__}: {exc}",
        })
        return 2
    if target.process_name.casefold() == EXPECTED_GAME_PROCESS.casefold():
        _emit(output, {
            **base,
            "result": "refused",
            "reason": "smoke_tool_requires_non_game_test_window",
            "target_process_id": target.process_id,
            "target_process_name": target.process_name,
        })
        return 2
    try:
        api = api_factory()
        integrity = dict(api.process_integrity_diagnostics(target.process_id))
        deadline = clock() + args.delay_seconds
        while clock() < deadline:
            if api.panic_pressed(VIRTUAL_KEYS["F12"]):
                _emit(output, {**base, "result": "panic_stop", "integrity": integrity})
                return 3
            sleep(min(0.05, max(0.0, deadline - clock())))
        snapshot = api.inspect_window(target.window_handle)
        rejection = None
        if not snapshot.exists:
            rejection = "target_window_invalid"
        elif not snapshot.visible:
            rejection = "target_window_not_visible"
        elif snapshot.minimized:
            rejection = "target_window_minimized"
        elif snapshot.title != target.window_title:
            rejection = "window_title_changed"
        elif snapshot.process_id != target.process_id:
            rejection = "target_process_changed"
        elif snapshot.foreground_hwnd != target.window_handle:
            rejection = "foreground_window_mismatch"
        if rejection:
            _emit(output, {
                **base,
                "result": "refused",
                "reason": rejection,
                "target_hwnd": target.window_handle,
                "foreground_hwnd": snapshot.foreground_hwnd,
                "integrity": integrity,
            })
            return 2
        if api.panic_pressed(VIRTUAL_KEYS["F12"]):
            _emit(output, {**base, "result": "panic_stop", "integrity": integrity})
            return 3
        down = api.send_key_event(
            VIRTUAL_KEYS[args.key], key_up=False, input_mode=args.input_mode
        )
        calls = [down]
        if down.return_count == 1:
            sleep(0.04)
            calls.append(api.send_key_event(
                VIRTUAL_KEYS[args.key], key_up=True, input_mode=args.input_mode
            ))
        emitted = len(calls) == 2 and all(item.return_count == 1 for item in calls)
        _emit(output, {
            **base,
            "result": "os_input_emitted" if emitted else "sendinput_failed",
            "target_hwnd": target.window_handle,
            "foreground_hwnd": snapshot.foreground_hwnd,
            "target_process_id": target.process_id,
            "target_process_name": target.process_name,
            "sendinput_calls": [asdict(item) for item in calls],
            "os_input_emitted": emitted,
            # A human visual check of Notepad is still required; the tool does
            # not treat a Win32 return count as application acknowledgement.
            "action_applied": False,
            "visual_acknowledgement_required": True,
            "integrity": integrity,
        })
        return 0 if emitted else 2
    except Exception as exc:
        _emit(output, {
            **base,
            "result": "failed",
            "reason": "smoke_test_exception",
            "error": f"{type(exc).__name__}: {exc}",
        })
        return 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_live_detect_only as live_cli
from src.fishing_v2.live.window_resolver import (
    ResolvedWindowTarget,
    WindowResolutionError,
    format_window_candidates,
    resolve_window_target,
)
from src.screen_capture import CaptureRegion, WindowInfo
from tools.run_live_detect_only import parse_args


def _window(
    hwnd: int,
    *,
    title: str = "黑色沙漠 - 525411",
    pid: int = 100,
    process: str = "BlackDesert64.exe",
    visible: bool = True,
    minimized: bool = False,
    width: int = 2560,
    height: int = 1440,
) -> WindowInfo:
    return WindowInfo(
        hwnd,
        title,
        pid,
        process,
        CaptureRegion(0, 0, width, height, hwnd, title),
        visible,
        minimized,
    )


@pytest.mark.parametrize("process_name", ["BlackDesert64", "BlackDesert64.exe"])
def test_auto_resolver_accepts_process_name_with_or_without_exe(
    process_name: str,
) -> None:
    target = resolve_window_target(
        process_name=process_name,
        title_prefix="黑色沙漠",
        enumerator=lambda: (_window(101),),
    )
    assert target.window_handle == 101
    assert target.process_id == 100
    assert target.window_title == "黑色沙漠 - 525411"
    assert target.resolution_mode == "process_name"


def test_auto_resolver_allows_title_suffix_to_change_between_runs() -> None:
    first = resolve_window_target(
        process_name="BlackDesert64",
        title_prefix="黑色沙漠",
        enumerator=lambda: (_window(101, title="黑色沙漠 - 525411"),),
    )
    second = resolve_window_target(
        process_name="BlackDesert64",
        title_prefix="黑色沙漠",
        enumerator=lambda: (_window(101, title="黑色沙漠 - 999999"),),
    )
    assert first.window_handle == second.window_handle == 101
    assert second.window_title == "黑色沙漠 - 999999"


def test_auto_resolver_filters_hidden_minimized_mismatch_and_invalid_client() -> None:
    selected = _window(105)
    target = resolve_window_target(
        process_name="BlackDesert64.exe",
        title_prefix="黑色沙漠",
        enumerator=lambda: (
            _window(101, visible=False),
            _window(102, minimized=True),
            _window(103, process="notepad.exe"),
            _window(104, width=0),
            _window(106, title=""),
            selected,
        ),
    )
    assert target.window_info == selected


def test_auto_resolver_fails_closed_with_zero_candidates() -> None:
    with pytest.raises(WindowResolutionError) as exc:
        resolve_window_target(
            process_name="BlackDesert64",
            title_prefix="黑色沙漠",
            enumerator=lambda: (_window(101, process="notepad.exe"),),
        )
    assert exc.value.reason == "window_candidate_not_found"
    assert exc.value.candidates == ()
    assert format_window_candidates(exc.value.candidates) == "  (none)"


def test_auto_resolver_fails_closed_and_lists_multiple_candidates() -> None:
    candidates = (_window(101), _window(202, pid=200))
    with pytest.raises(WindowResolutionError) as exc:
        resolve_window_target(
            process_name="BlackDesert64",
            title_prefix="黑色沙漠",
            enumerator=lambda: candidates,
        )
    assert exc.value.reason == "window_candidate_ambiguous"
    listing = format_window_candidates(exc.value.candidates)
    assert "PID=100 HWND=101" in listing
    assert "PID=200 HWND=202" in listing
    assert "client=2560x1440" in listing


def test_exact_title_mode_uses_existing_exact_resolver() -> None:
    calls: list[tuple[str, str | None]] = []

    def exact(title: str, *, expected_process_name: str | None) -> WindowInfo:
        calls.append((title, expected_process_name))
        return _window(303, title=title)

    target = resolve_window_target(
        exact_title="黑色沙漠 - 525411",
        expected_process_name="BlackDesert64",
        exact_resolver=exact,
    )
    assert target.window_handle == 303
    assert target.resolution_mode == "exact_title"
    assert calls == [("黑色沙漠 - 525411", "BlackDesert64")]


def test_cli_requires_one_resolution_mode_and_prefix_belongs_to_auto() -> None:
    auto = parse_args([
        "--process-name",
        "BlackDesert64",
        "--window-title-prefix",
        "黑色沙漠",
    ])
    assert auto.process_name == "BlackDesert64"
    assert auto.window_title is None
    exact = parse_args(["--window-title", "黑色沙漠 - 525411"])
    assert exact.window_title == "黑色沙漠 - 525411"
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args([
            "--window-title",
            "黑色沙漠 - 525411",
            "--window-title-prefix",
            "黑色沙漠",
        ])


@pytest.mark.parametrize(
    ("reason", "candidate_count"),
    [
        ("window_candidate_not_found", 0),
        ("window_candidate_ambiguous", 2),
    ],
)
def test_cli_resolution_failure_stops_before_capture_or_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
    candidate_count: int,
) -> None:
    candidates = tuple(_window(index + 100) for index in range(candidate_count))
    monkeypatch.setattr(
        live_cli,
        "resolve_window_target",
        lambda **_kwargs: (_ for _ in ()).throw(
            WindowResolutionError(reason, "resolution failed", candidates)
        ),
    )
    forbidden: list[str] = []
    monkeypatch.setattr(
        live_cli,
        "create_live_capture_session",
        lambda **_kwargs: forbidden.append("capture"),
    )
    monkeypatch.setattr(
        live_cli,
        "load_prompt_bundle",
        lambda *_args: forbidden.append("bundle"),
    )
    monkeypatch.setattr(
        live_cli.sys,
        "argv",
        [
            "run_live_detect_only.py",
            "--process-name",
            "BlackDesert64",
            "--window-title-prefix",
            "黑色沙漠",
        ],
    )

    assert live_cli.main() == 2
    assert forbidden == []
    error = capsys.readouterr().err
    assert "WINDOW RESOLUTION FAILED" in error
    assert "Candidates:" in error
    if candidate_count:
        assert "PID=100" in error
    else:
        assert "(none)" in error


def test_cli_passes_resolved_identity_to_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    info = _window(4242, pid=5151, title="黑色沙漠 - 999999")
    target = ResolvedWindowTarget(
        info,
        "process_name",
        "BlackDesert64",
        "黑色沙漠",
    )
    capture_calls: list[dict] = []
    runtime_calls: list[dict] = []

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            runtime_calls.append(kwargs)

        def run(self) -> dict:
            return {
                "result": "completed",
                "capture_backend": "mss-region",
                "capture_fallback_used": False,
                "evidence_mode": "minimal",
                "video_path": None,
                "completed_cycles": 0,
                "actions_applied": 0,
            }

    monkeypatch.setattr(
        live_cli, "resolve_window_target", lambda **_kwargs: target
    )
    monkeypatch.setattr(
        live_cli,
        "load_prompt_bundle",
        lambda _path: SimpleNamespace(bundle_version="test"),
    )
    monkeypatch.setattr(
        live_cli,
        "LiveSessionLogger",
        lambda *_args, **_kwargs: SimpleNamespace(path=tmp_path / "session"),
    )
    monkeypatch.setattr(
        live_cli,
        "create_live_capture_session",
        lambda **kwargs: capture_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(live_cli, "LiveDetectOnlyRuntime", FakeRuntime)
    monkeypatch.setattr(
        live_cli.sys,
        "argv",
        [
            "run_live_detect_only.py",
            "--process-name",
            "BlackDesert64.exe",
            "--window-title-prefix",
            "黑色沙漠",
            "--no-overlay",
        ],
    )

    assert live_cli.main() == 0
    assert len(capture_calls) == 1
    assert capture_calls[0]["window_title"] == target.window_title
    assert capture_calls[0]["resolved_window"] == info
    assert capture_calls[0]["expected_title_prefix"] == "黑色沙漠"
    assert capture_calls[0]["window_resolution_mode"] == "process_name"
    assert runtime_calls[0]["capture"] is not None

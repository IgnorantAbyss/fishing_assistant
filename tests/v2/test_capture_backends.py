from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.fishing_v2.live.capture_backends import (
    CAPTURE_BACKENDS,
    MSS_REGION_BACKEND,
    WINDOWS_GRAPHICS_CAPTURE_BACKEND,
    CapturedSurface,
    ExplicitFallbackCaptureSession,
    WindowsGraphicsCaptureSession,
    WindowsGraphicsCaptureUnavailable,
    create_live_capture_session,
)
from src.screen_capture import CaptureRegion, WindowInfo, resolve_exact_window
from tools.run_live_detect_only import parse_args


ROOT = Path(__file__).resolve().parents[2]


def _window_info(
    *, hwnd: int = 4242, title: str = "黑色沙漠 - 524983", width: int = 2, height: int = 1
) -> WindowInfo:
    return WindowInfo(
        hwnd,
        title,
        5151,
        "BlackDesert64",
        CaptureRegion(100, 200, width, height, hwnd, title),
    )


class MockWGCProvider:
    def __init__(self, hwnd: int, surface: CapturedSurface) -> None:
        self.capture_item_hwnd = hwnd
        self.surface = surface
        self.opened_with: tuple[int, int] | None = None
        self.valid = True
        self.closed = False

    def open(self, client_size: tuple[int, int]) -> None:
        self.opened_with = client_size

    def capture(self) -> CapturedSurface:
        return self.surface

    def is_valid(self) -> bool:
        return self.valid

    def close(self) -> None:
        self.closed = True


def _session(
    surface: CapturedSurface,
    *,
    info: WindowInfo | None = None,
    inspector=None,
) -> tuple[WindowsGraphicsCaptureSession, MockWGCProvider]:
    info = info or _window_info()
    provider = MockWGCProvider(info.window_handle, surface)

    def lookup(title: str, *, expected_process_name: str) -> WindowInfo:
        assert title == info.window_title
        assert expected_process_name == "BlackDesert64"
        return info

    session = WindowsGraphicsCaptureSession(
        window_title=info.window_title,
        expected_client_size=(info.client_region.width, info.client_region.height),
        provider_factory=lambda hwnd: provider,
        window_lookup=lookup,
        window_inspector=inspector or (lambda *_args, **_kwargs: info),
    )
    return session, provider


def test_capture_backend_cli_is_explicit_and_validated() -> None:
    default = parse_args(["--window-title", "exact-title"])
    assert default.capture_backend == MSS_REGION_BACKEND
    assert default.allow_mss_fallback is False
    assert default.evidence_mode == "minimal"
    assert default.evidence_video_fps == 10.0
    selected = parse_args([
        "--window-title", "exact-title",
        "--capture-backend", WINDOWS_GRAPHICS_CAPTURE_BACKEND,
        "--allow-mss-fallback",
        "--evidence-mode", "diagnostic",
        "--evidence-video-fps", "12",
        "--max-completed-cycles", "3",
    ])
    assert selected.capture_backend == WINDOWS_GRAPHICS_CAPTURE_BACKEND
    assert selected.allow_mss_fallback is True
    assert selected.evidence_mode == "diagnostic"
    assert selected.evidence_video_fps == 12.0
    assert selected.max_completed_cycles == 3
    assert CAPTURE_BACKENDS == ("mss-region", "windows-graphics-capture")
    with pytest.raises(SystemExit):
        parse_args(["--window-title", "exact-title", "--capture-backend", "monitor"])
    with pytest.raises(SystemExit):
        parse_args(["--window-title", "exact-title", "--evidence-mode", "dense"])


def test_exact_title_lookup_never_uses_similar_window() -> None:
    exact = "黑色沙漠 - 524983"
    handles = {exact: 101, "黑色沙漠 - 999999": 202}
    seen: list[str] = []

    def lookup(title: str) -> int:
        seen.append(title)
        return handles.get(title, 0)

    def inspect(hwnd: int, **_kwargs) -> WindowInfo:
        assert hwnd == 101
        return _window_info(hwnd=hwnd, title=exact)

    result = resolve_exact_window(
        exact,
        expected_process_name="BlackDesert64",
        handle_lookup=lookup,
        inspector=inspect,
    )
    assert result.window_handle == 101
    assert seen == [exact]


@pytest.mark.parametrize(
    ("order", "raw", "expected"),
    [
        ("BGRA", [[[11, 22, 33, 44], [55, 66, 77, 88]]], [[[11, 22, 33], [55, 66, 77]]]),
        ("RGBA", [[[33, 22, 11, 44], [77, 66, 55, 88]]], [[[11, 22, 33], [55, 66, 77]]]),
    ],
)
def test_wgc_adapter_removes_alpha_once_and_returns_canonical_bgr(
    order: str, raw: list[list[list[int]]], expected: list[list[list[int]]]
) -> None:
    session, provider = _session(CapturedSurface(np.asarray(raw, dtype=np.uint8), order))
    info = session.open()
    frame = session.capture()
    assert provider.capture_item_hwnd == info.window_handle
    assert provider.opened_with == (2, 1)
    assert frame.tolist() == expected
    assert frame.dtype == np.uint8
    assert frame.shape == (1, 2, 3)
    assert frame.flags.c_contiguous
    diagnostics = session.diagnostics()
    assert diagnostics["raw_capture_shape"] == [1, 2, 4]
    assert diagnostics["converted_shape"] == [1, 2, 3]
    assert diagnostics["channel_order"] == "BGR"
    assert len(diagnostics["frame_sha256"]) == 64


def test_wgc_rejects_wrong_client_or_surface_size_without_resize() -> None:
    wrong_client = _window_info(width=1280, height=720)
    provider = MockWGCProvider(
        wrong_client.window_handle,
        CapturedSurface(np.zeros((720, 1280, 4), dtype=np.uint8)),
    )
    session = WindowsGraphicsCaptureSession(
        window_title=wrong_client.window_title,
        provider_factory=lambda _hwnd: provider,
        window_lookup=lambda *_args, **_kwargs: wrong_client,
    )
    with pytest.raises(RuntimeError, match="client size 1280x720"):
        session.open()

    info = _window_info()
    mismatched, _ = _session(CapturedSurface(np.zeros((2, 2, 4), dtype=np.uint8)), info=info)
    mismatched.open()
    with pytest.raises(RuntimeError, match="returned 2x2"):
        mismatched.capture()


def test_wgc_provider_must_bind_the_same_exact_hwnd() -> None:
    info = _window_info()
    provider = MockWGCProvider(
        info.window_handle + 1,
        CapturedSurface(np.zeros((1, 2, 4), dtype=np.uint8)),
    )
    session = WindowsGraphicsCaptureSession(
        window_title=info.window_title,
        expected_client_size=(2, 1),
        provider_factory=lambda _hwnd: provider,
        window_lookup=lambda *_args, **_kwargs: info,
    )
    with pytest.raises(RuntimeError, match="different HWND"):
        session.open()
    assert provider.closed is True


def test_wgc_stops_when_original_hwnd_disappears_or_capture_item_is_invalid() -> None:
    surface = CapturedSurface(np.zeros((1, 2, 4), dtype=np.uint8))
    missing, _ = _session(
        surface,
        inspector=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Window HWND is no longer valid")
        ),
    )
    missing.open()
    with pytest.raises(RuntimeError, match="no longer valid"):
        missing.capture()

    invalid, provider = _session(surface)
    invalid.open()
    provider.valid = False
    with pytest.raises(RuntimeError, match="capture item is invalid"):
        invalid.capture()


class StubSession:
    def __init__(self, backend_name: str, *, failure: Exception | None = None) -> None:
        self.backend_name = backend_name
        self.failure = failure
        self.opened = False
        self.closed = False

    def open(self):
        if self.failure:
            raise self.failure
        self.opened = True
        return self.backend_name

    def capture(self):
        return np.zeros((1, 2, 3), dtype=np.uint8)

    def is_foreground(self):
        return True

    def diagnostics(self):
        return {"backend": self.backend_name}

    def close(self):
        self.closed = True


def test_wgc_initialization_never_silently_falls_back() -> None:
    failure = WindowsGraphicsCaptureUnavailable("binding unavailable")
    primary = StubSession(WINDOWS_GRAPHICS_CAPTURE_BACKEND, failure=failure)
    fallback = StubSession(MSS_REGION_BACKEND)
    blocked = ExplicitFallbackCaptureSession(primary, fallback, allowed=False)
    with pytest.raises(WindowsGraphicsCaptureUnavailable, match="unavailable"):
        blocked.open()
    assert fallback.opened is False

    primary = StubSession(WINDOWS_GRAPHICS_CAPTURE_BACKEND, failure=failure)
    fallback = StubSession(MSS_REGION_BACKEND)
    allowed = ExplicitFallbackCaptureSession(primary, fallback, allowed=True)
    assert allowed.open() == MSS_REGION_BACKEND
    assert fallback.opened is True
    assert allowed.diagnostics()["fallback_used"] is True
    assert "binding unavailable" in allowed.diagnostics()["fallback_reason"]


def test_factory_rejects_unknown_backend_before_capture() -> None:
    with pytest.raises(ValueError, match="Unsupported capture backend"):
        create_live_capture_session(backend="monitor", window_title="anything")


def test_capture_backend_layer_imports_no_input_or_runtime_decision_modules() -> None:
    path = ROOT / "src" / "fishing_v2" / "live" / "capture_backends.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.lower())
    forbidden = {
        "pyautogui", "pydirectinput", "pynput", "keyboard",
        "src.fishing_v2.fusion.observation_fusion",
        "src.fishing_v2.runtime.fishing_fsm",
        "src.fishing_v2.perception.prototype_prompt_observer",
    }
    assert imported.isdisjoint(forbidden)
    assert "ground_truth" not in path.read_text(encoding="utf-8").lower()

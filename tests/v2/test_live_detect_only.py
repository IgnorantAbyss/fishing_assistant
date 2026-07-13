import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import GetObservation, HookObservation, PressObservation
from src.fishing_v2.live.live_detect_only import (
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
    LivePreflightError,
    WouldFireDeduplicator,
    validate_emit_actions,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger, create_live_session_directory
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts" / "prompt_observer" / "prototype_v1"
CONFIG = ROOT / "config" / "fishing_v2.yaml"


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0.001)


class MockCapture:
    def __init__(self, frame: np.ndarray, failure: BaseException | None = None) -> None:
        self.frame = frame
        self.failure = failure
        self.calls = 0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def capture(self) -> np.ndarray:
        self.calls += 1
        if self.calls > 1 and self.failure is not None:
            raise self.failure
        return self.frame

    def is_foreground(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class NullHookDetector:
    def observe(self, _frame, context):
        return HookObservation(False, 0.0, context.frame_index, context.timestamp)


class NullPressDetector:
    def observe(self, _frame, context):
        return PressObservation(False, 0.0, context.frame_index, context.timestamp)


class NullGetDetector:
    def observe(self, _frame, context):
        return GetObservation(False, 0.0, context.frame_index, context.timestamp)


@pytest.fixture(scope="module")
def supported_frame() -> np.ndarray:
    return np.zeros((1440, 2560, 3), dtype=np.uint8)


def _runtime(tmp_path: Path, capture: MockCapture, clock: FakeClock) -> LiveDetectOnlyRuntime:
    return LiveDetectOnlyRuntime(
        config_path=CONFIG,
        prompt_bundle=load_prompt_bundle(BUNDLE),
        capture=capture,
        logger=LiveSessionLogger(tmp_path, bundle_version="test-bundle"),
        live_config=LiveDetectOnlyConfig(
            duration_seconds=1.0,
            max_fps=25.0,
            show_overlay=False,
            save_transition_frames=False,
        ),
        emit_actions=False,
        hook_detector=NullHookDetector(),
        press_detector=NullPressDetector(),
        get_detector=NullGetDetector(),
        clock=clock,
        sleep=clock.sleep,
    )


def test_emit_actions_true_is_refused_before_capture_initialization() -> None:
    with pytest.raises(LivePreflightError, match="forbidden"):
        validate_emit_actions(True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_live_detect_only.py"),
            "--window-title", "must-not-be-opened",
            "--emit-actions", "true",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "REFUSED" in result.stderr


def test_wrong_resolution_fails_preflight_and_closes_capture(tmp_path: Path) -> None:
    capture = MockCapture(np.zeros((720, 1280, 3), dtype=np.uint8))
    summary = _runtime(tmp_path, capture, FakeClock()).run(max_frames=1)
    assert summary["result"] == "preflight_failed"
    assert summary["actions_applied"] == 0
    assert capture.closed is True


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("capture disconnected"), "safe_stop_capture_failure"),
        (KeyboardInterrupt(), "interrupted_by_user"),
    ],
)
def test_capture_failure_and_ctrl_c_stop_safely(
    tmp_path: Path,
    supported_frame: np.ndarray,
    failure: BaseException,
    expected: str,
) -> None:
    capture = MockCapture(supported_frame, failure)
    summary = _runtime(tmp_path, capture, FakeClock()).run(max_frames=2)
    assert summary["result"] == expected
    assert summary["actions_applied"] == 0
    assert capture.closed is True


def test_mock_live_frame_uses_no_sink_and_never_applies_action(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    capture = MockCapture(supported_frame)
    runtime = _runtime(tmp_path, capture, FakeClock())
    assert runtime.controller.action_sink is None
    summary = runtime.run(max_frames=1)
    assert summary["result"] == "completed"
    assert summary["actions_applied"] == 0
    assert summary["emit_actions"] is False
    assert summary["action_sink"] is None
    saved = json.loads((runtime.logger.path / "session_summary.json").read_text(encoding="utf-8"))
    assert saved["actions_applied"] == 0


def test_would_fire_deduplicates_repeated_proposals_per_cycle() -> None:
    tracker = WouldFireDeduplicator()
    request = ActionRequest(ActionIntent.HOOK_ACTION, 0.96, "safe crossing")
    kwargs = {
        "safety_reason": "action_emission_disabled",
        "timestamp": 1.0,
        "runtime_state": "HOOK",
        "prompt_evidence": {},
        "specialized_evidence": {},
    }
    assert tracker.observe(request, frame_index=10, **kwargs) is not None
    assert tracker.observe(request, frame_index=11, **kwargs) is None
    assert tracker.raw_proposals == {"HOOK_ACTION": 2}
    assert tracker.unique_events == {"WOULD_HOOK_ACTION": 1}
    tracker.finish_cycle()
    assert tracker.observe(request, frame_index=20, **kwargs) is not None
    assert tracker.unique_events == {"WOULD_HOOK_ACTION": 2}


def test_live_session_directory_never_overwrites(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 13, 1, 2, 3, tzinfo=timezone.utc)
    first = create_live_session_directory(tmp_path, timestamp=timestamp)
    second = create_live_session_directory(tmp_path, timestamp=timestamp)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_live_runtime_imports_no_input_writer_and_never_loads_ground_truth() -> None:
    paths = [
        ROOT / "src" / "fishing_v2" / "live" / "live_detect_only.py",
        ROOT / "src" / "fishing_v2" / "live" / "session_logger.py",
        ROOT / "src" / "screen_capture.py",
        ROOT / "tools" / "run_live_detect_only.py",
    ]
    imported = set()
    attributes = set()
    combined = ""
    for path in paths:
        source = path.read_text(encoding="utf-8")
        combined += source.lower()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0].lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0].lower())
            elif isinstance(node, ast.Attribute):
                attributes.add(node.attr.lower())
    assert imported.isdisjoint({"pyautogui", "pynput", "keyboard"})
    assert "sendinput" not in attributes
    assert "ground_truth.yaml" not in combined
    assert "prompt_ground_truth" not in combined

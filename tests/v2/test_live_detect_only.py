import ast
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.live.live_detect_only import (
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
    LivePreflightError,
    WouldFireDeduplicator,
    validate_emit_actions,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger, create_live_session_directory
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle
from src.fishing_v2.perception.prototype_prompt_observer import (
    PrototypePromptModel,
    extract_prompt_feature,
    validate_prompt_input,
)
from src.screen_capture import mss_bgra_to_bgr


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
    def __init__(
        self,
        frame: np.ndarray,
        failure: BaseException | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        self.frame = frame
        self.failure = failure
        self._diagnostics = diagnostics or {}
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

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)

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


def _runtime(
    tmp_path: Path,
    capture: MockCapture,
    clock: FakeClock,
    *,
    duration_seconds: float = 1.0,
    evidence_mode: str = "minimal",
    evidence_recorder=None,
) -> LiveDetectOnlyRuntime:
    return LiveDetectOnlyRuntime(
        config_path=CONFIG,
        prompt_bundle=load_prompt_bundle(BUNDLE),
        capture=capture,
        logger=LiveSessionLogger(tmp_path, bundle_version="test-bundle"),
        live_config=LiveDetectOnlyConfig(
            duration_seconds=duration_seconds,
            max_fps=25.0,
            show_overlay=False,
            save_transition_frames=False,
            evidence_mode=evidence_mode,
        ),
        emit_actions=False,
        hook_detector=NullHookDetector(),
        press_detector=NullPressDetector(),
        get_detector=NullGetDetector(),
        evidence_recorder=evidence_recorder,
        clock=clock,
        sleep=clock.sleep,
    )


class StubEvidenceRecorder:
    def __init__(self) -> None:
        self.video_frames: list[tuple[int, float, tuple[int, ...]]] = []
        self.roi_frames: list[dict[str, object]] = []
        self.events: list[str] = []
        self.finalized = False

    def record_frame(self, frame, *, capture_frame_index, timestamp):
        self.video_frames.append((capture_frame_index, timestamp, frame.shape))
        return True

    def record_detector_evidence(self, _frame, **kwargs):
        self.roi_frames.append(kwargs)

    def mark_event(self, event_type, _payload):
        self.events.append(event_type)

    def finalize(self):
        self.finalized = True
        return {
            "evidence_mode": "diagnostic",
            "video_path": "diagnostic_evidence/session_capture.mp4",
            "video_frame_count": len(self.video_frames),
            "video_fps": 10.0,
            "first_timestamp": self.video_frames[0][1] if self.video_frames else None,
            "last_timestamp": self.video_frames[-1][1] if self.video_frames else None,
            "dropped_video_frames": 0,
            "roi_evidence_counts_by_episode": {"1": {"prompt": len(self.roi_frames)}},
            "has_evidence_gaps": False,
            "evidence_gap_intervals": [],
        }


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
    assert not (runtime.logger.path / "diagnostic_evidence").exists()


def test_diagnostic_mode_records_video_and_roi_without_would_fire(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    recorder = StubEvidenceRecorder()
    runtime = _runtime(
        tmp_path,
        MockCapture(supported_frame),
        FakeClock(),
        evidence_mode="diagnostic",
        evidence_recorder=recorder,
    )
    summary = runtime.run(max_frames=3)
    assert summary["result"] == "completed"
    assert summary["video_frame_count"] == 3
    assert recorder.video_frames[0][2] == (1440, 2560, 3)
    assert recorder.roi_frames
    assert set(recorder.roi_frames[0]["roi_bounds"]) == {"prompt", "hook", "press", "get"}
    assert summary["unique_would_fire"] == {}
    assert summary["actions_applied"] == 0
    assert recorder.finalized is True


def test_diagnostic_video_finalizes_after_ctrl_c(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    class DelayedInterruptCapture(MockCapture):
        def capture(self):
            self.calls += 1
            if self.calls > 2:
                raise KeyboardInterrupt()
            return self.frame

    recorder = StubEvidenceRecorder()
    runtime = _runtime(
        tmp_path,
        DelayedInterruptCapture(supported_frame),
        FakeClock(),
        evidence_mode="diagnostic",
        evidence_recorder=recorder,
    )
    summary = runtime.run(max_frames=5)
    assert summary["result"] == "interrupted_by_user"
    assert summary["video_frame_count"] == 1
    assert recorder.finalized is True
    assert summary["actions_applied"] == 0


def test_explicit_capture_fallback_is_recorded_as_session_warning(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    capture = MockCapture(supported_frame, diagnostics={
        "backend": "mss-region",
        "requested_backend": "windows-graphics-capture",
        "fallback_used": True,
        "fallback_reason": "WindowsGraphicsCaptureUnavailable: binding unavailable",
        "overlay_capture_warning": False,
    })
    runtime = _runtime(tmp_path, capture, FakeClock())
    summary = runtime.run(max_frames=1)
    saved = json.loads((runtime.logger.path / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["capture_backend"] == "mss-region"
    assert summary["capture_fallback_used"] is True
    assert any("binding unavailable" in warning for warning in saved["warnings"])
    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event_type"] == "capture_backend_fallback" for item in events)


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
        ROOT / "src" / "fishing_v2" / "live" / "capture_backends.py",
        ROOT / "src" / "fishing_v2" / "live" / "diagnostic_evidence.py",
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


def test_mss_bgra_conversion_preserves_bgr_order_and_canonical_contract() -> None:
    bgra = np.asarray([[[11, 22, 33, 44], [55, 66, 77, 88]]], dtype=np.uint8)
    bgr = mss_bgra_to_bgr(bgra)
    assert bgr.tolist() == [[[11, 22, 33], [55, 66, 77]]]
    assert bgr.dtype == np.uint8
    assert bgr.flags.c_contiguous
    validate_prompt_input(bgr)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 4, 3), dtype=np.uint8)[:, ::2],
    ],
)
def test_prompt_input_contract_rejects_noncanonical_frames(invalid: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_prompt_input(invalid)


def test_confirmed_saved_live_bite_variant_is_ready_without_margin_relaxation() -> None:
    import cv2

    crop = cv2.imread(str(
        ROOT / "assets" / "reference" / "prompt" / "live_ready_bite_20260713_000035.png"
    ))
    loaded = load_prompt_bundle(BUNDLE)
    feature = extract_prompt_feature(crop)
    base_model = PrototypePromptModel(
        tuple(item for item in loaded.model.prototypes if item.source_session in loaded.model.training_sessions),
        loaded.model.class_thresholds,
        loaded.model.ambiguity_threshold,
        loaded.model.training_sessions,
        loaded.model.calibration_sessions,
        loaded.model.idle_stability_frames,
    )
    baseline = base_model.predict_feature(feature)
    assert baseline.predicted_label == "UNKNOWN"
    assert baseline.prototype_id == "HOOK_INSTRUCTION:session_20260709_192315:000465"
    assert baseline.second_label == "PRESS_INSTRUCTION"
    corrected = loaded.model.predict_feature(feature)
    assert corrected.predicted_label == "READY_BITE"
    assert corrected.similarity == pytest.approx(1.0, abs=1e-6)
    assert corrected.ambiguity_margin >= loaded.model.ambiguity_threshold

    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    x1, y1, x2, y2 = loaded.roi.pixel
    frame[y1:y2, x1:x2] = crop
    labels = [
        loaded.observer.observe(frame, FrameContext(index, index * 0.2)).kind.value
        for index in range(1, 31)
    ]
    assert labels == ["READY_BITE"] * 30
    assert "IDLE_CAST" not in labels
    assert "HOOK_INSTRUCTION" not in labels
    assert "PRESS_INSTRUCTION" not in labels


def test_ready_holdout_startup_would_start_hook_once_and_never_cast(tmp_path: Path) -> None:
    import cv2

    holdout = cv2.imread(str(
        ROOT / "reports" / "fishing_v2" / "live_prompt_diagnostics"
        / "live_holdout_raw_roi.png"
    ))
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    loaded = load_prompt_bundle(BUNDLE)
    x1, y1, x2, y2 = loaded.roi.pixel
    frame[y1:y2, x1:x2] = holdout
    clock = FakeClock()
    runtime = _runtime(tmp_path, MockCapture(frame), clock, duration_seconds=6.0)
    summary = runtime.run(max_frames=150)
    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [item["event_type"] for item in events]
    assert summary["result"] == "completed"
    assert summary["unique_would_fire"].get("WOULD_START_HOOK") == 1
    assert summary["unique_would_fire"].get("WOULD_CAST", 0) == 0
    assert summary["raw_action_proposals"].get("CAST", 0) == 0
    assert summary["actions_applied"] == 0
    assert "SYNC_REQUIRED" not in event_types
    assert any(
        item["event_type"] == "prompt_label_change" and item["next_label"] == "READY_BITE"
        for item in events
    )


def test_live_runtime_logs_startup_and_sync_required_recovery_transitions(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    class TimelinePromptObserver:
        def __init__(self) -> None:
            self.calls = 0

        def observe(self, _frame, context):
            self.calls += 1
            kind = (
                PromptObservationKind.READY_BITE
                if self.calls <= 10
                else PromptObservationKind.WAITING_IN_PROGRESS
            )
            return PromptObservation(
                kind,
                0.998,
                {kind.value: 0.998},
                "saved_live_timeline",
                context.frame_index,
                context.timestamp,
                {},
            )

    runtime = _runtime(
        tmp_path,
        MockCapture(supported_frame),
        FakeClock(),
        duration_seconds=7.0,
    )
    runtime.prompt_bundle = replace(
        runtime.prompt_bundle,
        observer=TimelinePromptObserver(),
    )
    summary = runtime.run(max_frames=180)
    with runtime.logger.transitions_path.open(encoding="utf-8", newline="") as handle:
        transitions = list(csv.DictReader(handle))
    edges = [
        (row["previous_state"], row["next_state"], row["reason"])
        for row in transitions
    ]
    assert any(previous == "SYNCING" and next_state == "READY" for previous, next_state, _ in edges)
    assert (
        "READY",
        "SYNC_REQUIRED",
        "persistent_conflicting_or_illegal_evidence",
    ) in edges
    assert any(
        previous == "SYNC_REQUIRED"
        and next_state == "WAITING"
        and reason == "prompt_consensus_sync_recovery"
        for previous, next_state, reason in edges
    )

    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    types = [item["event_type"] for item in events]
    assert "sync_recovery_started" in types
    assert "sync_recovery_candidate" in types
    assert "sync_recovered" in types
    recovered = next(item for item in events if item["event_type"] == "sync_recovered")
    assert recovered["recovery_target"] == "WAITING"
    assert recovered["support_frames"] == 10
    assert recovered["action_intent"] == "NONE"
    assert recovered["action_applied"] is False
    recovery_frame = recovered["frame_index"]
    assert not any(
        item["frame_index"] == recovery_frame and item["event_type"].startswith("WOULD_")
        for item in events
        if "frame_index" in item
    )
    assert any(
        item["event_type"] == "detector_activation_change"
        and item["hook"] == "ARMED"
        and item["press"] == "ARMED"
        and item["get"] == "ARMED"
        for item in events
    )
    assert summary["actions_applied"] == 0
    assert runtime.controller.action_sink is None

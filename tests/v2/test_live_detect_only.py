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
    ResultBannerObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.live.live_detect_only import (
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
    LivePreflightError,
    WouldFireDeduplicator,
    validate_emit_actions,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger, create_live_session_directory
from src.fishing_v2.live.windows_action_sink import ActionIntegrityPreflightError
from src.fishing_v2.ports.action_sink import ActionExecutionResult
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
    emit_actions: bool = False,
    action_sink_name: str = "none",
    action_allowlist=(),
    action_sink_factory=None,
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
        emit_actions=emit_actions,
        action_sink_name=action_sink_name,
        action_allowlist=action_allowlist,
        action_sink_factory=action_sink_factory,
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


def test_emit_actions_true_requires_explicit_sink_before_capture_initialization() -> None:
    with pytest.raises(LivePreflightError, match="requires.*sendinput"):
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


def test_live_action_mode_accepts_staged_hook_action() -> None:
    validate_emit_actions(
        True,
        "sendinput",
        "CAST,START_HOOK,HOOK_ACTION,COLLECT",
    )


def test_live_action_mode_refuses_press_sequence() -> None:
    with pytest.raises(
        LivePreflightError,
        match="limited to CAST,START_HOOK,HOOK_ACTION,COLLECT",
    ):
        validate_emit_actions(
            True,
            "sendinput",
            "CAST,START_HOOK,HOOK_ACTION,COLLECT,PRESS_SEQUENCE",
        )


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


def test_emit_false_never_initializes_action_sink_factory(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    calls = []

    def forbidden_factory(**kwargs):
        calls.append(kwargs)
        raise AssertionError("action sink must not initialize")

    runtime = _runtime(
        tmp_path, MockCapture(supported_frame), FakeClock(),
        action_sink_factory=forbidden_factory,
    )
    summary = runtime.run(max_frames=1)
    assert summary["action_sink_type"] == "none"
    assert summary["emit_actions"] is False
    assert calls == []


def test_explicit_sendinput_mode_initializes_only_after_valid_target_preflight(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    created = []

    class FakeSink:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.apply_calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.apply_calls.append((request, context))
            raise AssertionError("one blank frame must not propose an action")

        def summary(self):
            return {"action_sink_type": "sendinput", "action_allowlist": ["COLLECT"]}

    def factory(**kwargs):
        sink = FakeSink(kwargs)
        created.append(sink)
        return sink

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title_prefix": "黑色沙漠",
        "window_resolution_mode": "process_name",
        "window_title": "黑色沙漠 - 525411",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path, capture, FakeClock(), emit_actions=True,
        action_sink_name="sendinput", action_allowlist="COLLECT",
        action_sink_factory=factory,
    )
    assert created == []
    summary = runtime.run(max_frames=1)
    assert len(created) == 1
    assert created[0].kwargs["target_hwnd"] == 4242
    assert created[0].kwargs["expected_title"] == "黑色沙漠 - 525411"
    assert created[0].kwargs["expected_title_prefix"] == "黑色沙漠"
    assert created[0].kwargs["window_resolution_mode"] == "process_name"
    assert created[0].kwargs["expected_process_id"] == 99
    assert created[0].apply_calls == []
    assert summary["action_sink_type"] == "sendinput"
    assert summary["actions_applied"] == 0


def test_integrity_mismatch_fails_live_preflight_before_capture_loop_or_input(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    factory_calls = []
    diagnostics = {
        "python_process": {
            "process_id": 1, "integrity_level": "medium",
            "integrity_rid": 8192, "elevated": False,
        },
        "target_process": {
            "process_id": 99, "integrity_level": "high",
            "integrity_rid": 12288, "elevated": True,
        },
        "suspected_integrity_mismatch": True,
    }

    def factory(**kwargs):
        factory_calls.append(kwargs)
        raise ActionIntegrityPreflightError(
            "integrity_mismatch",
            "Python process integrity is lower than target process integrity. "
            "Start the runtime manually from an elevated PowerShell.",
            diagnostics,
        )

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path, capture, FakeClock(), emit_actions=True,
        action_sink_name="sendinput",
        action_allowlist="CAST,START_HOOK,COLLECT",
        action_sink_factory=factory,
    )
    summary = runtime.run(max_frames=10)
    assert len(factory_calls) == 1
    assert capture.calls == 1
    assert summary["result"] == "preflight_failed"
    assert summary["preflight_passed"] is False
    assert summary["preflight_failure_reason"] == "integrity_mismatch"
    assert "elevated PowerShell" in summary["preflight_failure_message"]
    assert summary["python_integrity"]["integrity_level"] == "medium"
    assert summary["target_integrity"]["integrity_level"] == "high"
    assert summary["suspected_integrity_mismatch"] is True
    assert summary["actions_applied"] == 0
    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event_type"] == "preflight_failed" for item in events)


def test_collect_only_live_path_applies_one_stable_action_after_qualified_get(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    class VisibleGetDetector:
        def observe(self, _frame, context):
            return GetObservation(
                True, 0.99, context.frame_index, context.timestamp,
                evidence={"grid_cell_candidates": 12},
            )

    class CompleteFakeSink:
        def __init__(self, _kwargs):
            self.calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.calls.append((request, context))
            return ActionExecutionResult(
                context.action_id, request.intent.value, context.requested_at,
                context.requested_at, context.requested_at, True, True,
                2, 2, context.target_hwnd, context.target_hwnd,
                os_input_emitted=True,
            )

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": ["COLLECT"],
                "attempted_action_counts": {"COLLECT": len(self.calls)},
                "applied_action_counts": {"COLLECT": len(self.calls)},
            }

    created = []

    def factory(**kwargs):
        sink = CompleteFakeSink(kwargs)
        created.append(sink)
        return sink

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "黑色沙漠 - 525411",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path, capture, FakeClock(), duration_seconds=2.0,
        emit_actions=True, action_sink_name="sendinput",
        action_allowlist="COLLECT", action_sink_factory=factory,
    )
    runtime.get_detector = VisibleGetDetector()
    runtime.controller.evidence_qualifier.qualify_get(
        GetObservation(True, 0.99, 0, 0.0), DetectorActivationMode.BURST
    )
    runtime.fsm.force_state(RuntimeState.GET, 0.0, "test_get")
    summary = runtime.run(max_frames=12)
    assert len(created) == 1
    assert len(created[0].calls) == 1
    request, context = created[0].calls[0]
    assert request.intent == ActionIntent.COLLECT
    assert context.action_id == "get_episode:1:COLLECT:attempt:1"
    assert request.payload["elapsed_seconds"] >= 0.4
    assert summary["unique_would_fire"] == {"WOULD_COLLECT": 1}
    assert summary["actions_applied"] == 1
    assert summary["applied_action_counts"] == {"COLLECT": 1}
    assert summary["collect_attempt_counts"] == {"get_episode:1:COLLECT": 1}
    assert summary["collect_retry_counts"] == {"get_episode:1:COLLECT": 0}
    assert summary["collect_attempt_counts_by_get_episode"] == {
        "get_episode:1": 1
    }
    assert summary["collect_completed_count"] == 0
    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [item["event_type"] for item in events]
    assert "collect_retry_started" in event_names
    assert "collect_attempt_scheduled" in event_names
    assert "collect_attempt_started" in event_names
    assert "collect_attempt_emitted" in event_names
    assert "collect_attempt_waiting_ack" in event_names


def test_cast_collect_live_path_casts_once_then_waits_for_visual_ack(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    class CastThenWaitingObserver:
        def observe(self, _frame, context):
            kind = (
                PromptObservationKind.IDLE_CAST
                if context.frame_index < 48
                else PromptObservationKind.WAITING_IN_PROGRESS
            )
            return PromptObservation(
                kind, 0.99, {kind.value: 0.99}, "test",
                context.frame_index, context.timestamp,
            )

    class AbsentResultBanner:
        def observe(self, _frame, context):
            return ResultBannerObservation(
                False, 0.99, context.frame_index, context.timestamp,
                evidence={"reason": "absent"},
            )

    class CompleteFakeSink:
        def __init__(self, _kwargs):
            self.calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.calls.append((request, context))
            return ActionExecutionResult(
                context.action_id, request.intent.value, context.requested_at,
                context.requested_at, context.requested_at, True, True,
                2, 2, context.target_hwnd, context.target_hwnd,
                os_input_emitted=True,
            )

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": ["CAST", "COLLECT"],
                "attempted_action_counts": {"CAST": len(self.calls)},
                "applied_action_counts": {"CAST": len(self.calls)},
            }

    created = []

    def factory(**kwargs):
        sink = CompleteFakeSink(kwargs)
        created.append(sink)
        return sink

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path, capture, FakeClock(), duration_seconds=4.0,
        emit_actions=True, action_sink_name="sendinput",
        action_allowlist="cast, collect", action_sink_factory=factory,
    )
    runtime.prompt_bundle = replace(
        runtime.prompt_bundle, observer=CastThenWaitingObserver()
    )
    runtime.result_banner_observer = AbsentResultBanner()
    runtime.fsm.force_state(RuntimeState.RESULT_PENDING, 0.0, "test_result_pending")
    summary = runtime.run(max_frames=80)

    assert len(created) == 1
    assert len(created[0].calls) == 1
    request, context = created[0].calls[0]
    assert request.intent == ActionIntent.CAST
    assert context.action_id == "cast_opportunity:1:CAST"
    assert summary["cast_opportunity_count"] == 1
    assert summary["no_get_clearance_count"] == 1
    assert summary["cast_attempt_count"] == 1
    assert summary["cast_visual_acknowledged_count"] == 1
    assert summary["cast_timeout_count"] == 0
    assert summary["completed_cycles"] == 1
    assert summary["physical_get_episode_count"] == 0
    assert summary["raw_action_proposals"]["CAST"] > 0
    assert summary["unique_would_fire"] == {"WOULD_CAST": 1}
    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [item["event_type"] for item in events]
    assert event_names.count("cast_opportunity_started") == 1
    assert event_names.count("cast_attempt_emitted") == 1
    assert event_names.count("cast_visual_acknowledged") == 1
    blocked = [
        item for item in events
        if item["event_type"] == "cast_opportunity_blocked"
    ]
    assert blocked
    assert all("blockers" in item for item in blocked)
    assert all("get_presence_state" in item for item in blocked)
    assert all("result_banner_presence_state" in item for item in blocked)


def test_post_collect_window_runs_fresh_result_banner_confirmation_burst(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    class CountingAbsentBanner:
        def __init__(self):
            self.calls = []

        def observe(self, _frame, context):
            self.calls.append((context.frame_index, context.timestamp))
            return ResultBannerObservation(
                False, 0.99, context.frame_index, context.timestamp,
                evidence={"reason": "fresh_post_collect_absent"},
            )

    class NoInputSink:
        def poll_panic(self):
            return False

        def apply(self, _request, _context):
            raise AssertionError("confirmation burst must not emit input")

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": ["CAST", "COLLECT"],
                "panic_triggered": False,
            }

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path, capture, FakeClock(), duration_seconds=1.0,
        emit_actions=True, action_sink_name="sendinput",
        action_allowlist="CAST,COLLECT",
        action_sink_factory=lambda **_kwargs: NoInputSink(),
    )
    banner = CountingAbsentBanner()
    runtime.result_banner_observer = banner
    runtime.fsm.force_state(RuntimeState.IDLE, 0.0, "test_post_collect_idle")
    runtime.cast_clearance.observe(
        timestamp=-1.0,
        previous_state=RuntimeState.HOOK,
        current_state=RuntimeState.GET,
        prompt_kind=PromptObservationKind.UNKNOWN,
        prompt_frame_index=1,
        get_observation=GetObservation(True, 0.99, 1, -1.0),
        get_activation_mode=DetectorActivationMode.ACTIVE,
        result_banner=None,
        physical_get_episode_open=True,
        physical_get_episode_id="get_episode:1",
        physical_get_panel_visible=True,
    )
    runtime.cast_clearance.observe(
        timestamp=-0.5,
        previous_state=RuntimeState.GET,
        current_state=RuntimeState.COLLECT_PENDING,
        prompt_kind=PromptObservationKind.UNKNOWN,
        prompt_frame_index=2,
        get_observation=GetObservation(False, 0.99, 2, -0.5),
        get_activation_mode=DetectorActivationMode.ACTIVE,
        result_banner=None,
        physical_get_episode_open=False,
        physical_get_episode_id="get_episode:1",
        physical_get_episode_terminal=True,
        physical_get_panel_visible=False,
        collect_visual_acknowledged=True,
        collect_complete_emission_count=1,
        collect_terminal_reason="qualified_get_panel_stably_disappeared",
    )
    assert runtime.cast_clearance.post_collect_confirmation_required is True

    summary = runtime.run(max_frames=5)
    assert len(banner.calls) >= 2
    certificate = runtime.cast_clearance.status(
        banner.calls[-1][1]
    ).result_banner_absence_certificate
    assert certificate is not None
    assert certificate.source == "post_collect_confirmation"
    assert certificate.originating_get_episode_id == "get_episode:1"
    assert summary["actions_applied"] == 0


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


def test_live_qualified_hook_emits_exactly_once_and_arms_result_flow(
    tmp_path: Path,
    supported_frame: np.ndarray,
) -> None:
    class PersistentCrossedHookDetector:
        def observe(self, _frame, context):
            return HookObservation(
                True,
                0.99,
                context.frame_index,
                context.timestamp,
                fill_ratio=0.70,
                evidence={
                    "matched_features": [
                        "hook_bar_rect",
                        "bar_fill",
                    ],
                    "fallback_ratio_trustworthy": True,
                },
            )

    class CompleteHookSink:
        def __init__(self, _kwargs):
            self.calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.calls.append((request, context))
            return ActionExecutionResult(
                context.action_id,
                request.intent.value,
                context.requested_at,
                context.requested_at,
                context.requested_at,
                True,
                True,
                2,
                2,
                context.target_hwnd,
                context.target_hwnd,
                os_input_emitted=True,
            )

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": [
                    "CAST",
                    "COLLECT",
                    "HOOK_ACTION",
                    "START_HOOK",
                ],
                "attempted_action_counts": {
                    "HOOK_ACTION": len(self.calls),
                },
                "applied_action_counts": {
                    "HOOK_ACTION": len(self.calls),
                },
            }

    created = []

    def factory(**kwargs):
        sink = CompleteHookSink(kwargs)
        created.append(sink)
        return sink

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path,
        capture,
        FakeClock(),
        duration_seconds=0.8,
        emit_actions=True,
        action_sink_name="sendinput",
        action_allowlist="CAST,START_HOOK,HOOK_ACTION,COLLECT",
        action_sink_factory=factory,
    )
    runtime.hook_detector = PersistentCrossedHookDetector()
    runtime.fsm.force_state(RuntimeState.HOOK, 0.0, "test_hook")
    summary = runtime.run(max_frames=20)

    assert len(created) == 1
    assert len(created[0].calls) == 1
    request, context = created[0].calls[0]
    assert request.intent == ActionIntent.HOOK_ACTION
    assert context.runtime_state == RuntimeState.HOOK.value
    assert runtime.fsm.state == RuntimeState.RESULT_PENDING
    assert summary["unique_would_fire"] == {"WOULD_HOOK_ACTION": 1}
    assert summary["actions_applied"] == 1
    assert summary["detector_runs"]["press"] > 0
    assert summary["detector_runs"]["get"] > 0


def test_live_ready_burst_transitions_and_proposes_without_four_second_poll(
    tmp_path: Path, supported_frame: np.ndarray
) -> None:
    class ReadyObserver:
        def observe(self, _frame, context):
            return PromptObservation(
                PromptObservationKind.READY_BITE,
                0.99,
                {PromptObservationKind.READY_BITE.value: 0.99},
                "existing_prompt_observer",
                context.frame_index,
                context.timestamp,
                evidence={
                    "similarity": 0.99,
                    "ambiguity_margin": 0.3,
                    "rejection_reason": None,
                },
            )

    clock = FakeClock()
    runtime = _runtime(
        tmp_path, MockCapture(supported_frame), clock, duration_seconds=1.0
    )
    runtime.prompt_bundle = replace(runtime.prompt_bundle, observer=ReadyObserver())
    runtime.fsm.force_state(RuntimeState.WAITING, 0.0, "test_waiting")
    summary = runtime.run(max_frames=8)

    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    transition = next(
        item for item in events
        if item["event_type"] == "runtime_transition"
        and item["previous_state"] == "WAITING"
        and item["next_state"] == "READY"
    )
    would_start = next(
        item for item in events if item["event_type"] == "WOULD_START_HOOK"
    )
    confirmed = next(
        item for item in events
        if item["event_type"] == "ready_confirmation_burst_confirmed"
    )

    assert transition["timestamp"] <= 0.3
    assert would_start["timestamp"] - transition["timestamp"] < 0.1
    assert confirmed["action_intent"] == ActionIntent.NONE.value
    assert summary["unique_would_fire"]["WOULD_START_HOOK"] == 1
    assert summary["actions_applied"] == 0
    assert summary["action_sink"] is None


def test_live_stable_ready_emits_exactly_one_start_hook_and_waits_for_ack(
    tmp_path: Path,
    supported_frame: np.ndarray,
) -> None:
    class PersistentReadyObserver:
        def observe(self, _frame, context):
            return PromptObservation(
                PromptObservationKind.READY_BITE,
                0.99,
                {PromptObservationKind.READY_BITE.value: 0.99},
                "stable_ready_test",
                context.frame_index,
                context.timestamp,
                evidence={"rejection_reason": None},
            )

    class CompleteStartHookSink:
        def __init__(self, _kwargs):
            self.calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.calls.append((request, context))
            return ActionExecutionResult(
                context.action_id,
                request.intent.value,
                context.requested_at,
                context.requested_at,
                context.requested_at,
                True,
                True,
                2,
                2,
                context.target_hwnd,
                context.target_hwnd,
                os_input_emitted=True,
            )

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": ["CAST", "COLLECT", "START_HOOK"],
                "attempted_action_counts": {"START_HOOK": len(self.calls)},
                "applied_action_counts": {"START_HOOK": len(self.calls)},
            }

    created = []

    def factory(**kwargs):
        sink = CompleteStartHookSink(kwargs)
        created.append(sink)
        return sink

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path,
        capture,
        FakeClock(),
        duration_seconds=0.8,
        emit_actions=True,
        action_sink_name="sendinput",
        action_allowlist="CAST,START_HOOK,COLLECT",
        action_sink_factory=factory,
    )
    runtime.prompt_bundle = replace(
        runtime.prompt_bundle,
        observer=PersistentReadyObserver(),
    )
    runtime.fsm.force_state(RuntimeState.WAITING, 0.0, "test_waiting")
    summary = runtime.run(max_frames=20)

    assert len(created) == 1
    assert len(created[0].calls) == 1
    request, context = created[0].calls[0]
    assert request.intent == ActionIntent.START_HOOK
    assert context.runtime_state == RuntimeState.READY.value
    assert runtime.fsm.state == RuntimeState.HOOK_PENDING
    assert summary["unique_would_fire"] == {"WOULD_START_HOOK": 1}
    assert summary["actions_applied"] == 1
    with runtime.logger.transitions_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        transitions = list(csv.DictReader(handle))
    assert any(
        row["previous_state"] == RuntimeState.READY.value
        and row["next_state"] == RuntimeState.HOOK_PENDING.value
        and row["reason"] == "start_hook_action_applied"
        for row in transitions
    )


def test_live_failed_start_hook_emission_is_not_retried(
    tmp_path: Path,
    supported_frame: np.ndarray,
) -> None:
    class PersistentReadyObserver:
        def observe(self, _frame, context):
            return PromptObservation(
                PromptObservationKind.READY_BITE,
                0.99,
                {PromptObservationKind.READY_BITE.value: 0.99},
                "stable_ready_test",
                context.frame_index,
                context.timestamp,
            )

    class FailedStartHookSink:
        def __init__(self, _kwargs):
            self.calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.calls.append((request, context))
            return ActionExecutionResult(
                context.action_id,
                request.intent.value,
                context.requested_at,
                context.requested_at,
                context.requested_at,
                False,
                False,
                0,
                2,
                context.target_hwnd,
                context.target_hwnd,
                rejection_reason="sendinput_incomplete",
                os_input_emitted=False,
            )

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": ["START_HOOK"],
                "attempted_action_counts": {
                    "START_HOOK": len(self.calls),
                },
                "applied_action_counts": {},
            }

    created = []

    def factory(**kwargs):
        sink = FailedStartHookSink(kwargs)
        created.append(sink)
        return sink

    capture = MockCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path,
        capture,
        FakeClock(),
        duration_seconds=0.8,
        emit_actions=True,
        action_sink_name="sendinput",
        action_allowlist="START_HOOK",
        action_sink_factory=factory,
    )
    runtime.prompt_bundle = replace(
        runtime.prompt_bundle,
        observer=PersistentReadyObserver(),
    )
    runtime.fsm.force_state(RuntimeState.WAITING, 0.0, "test_waiting")
    summary = runtime.run(max_frames=20)

    assert len(created) == 1
    assert len(created[0].calls) == 1
    assert runtime.fsm.state == RuntimeState.READY
    assert summary["unique_would_fire"] == {"WOULD_START_HOOK": 1}
    assert summary["actions_applied"] == 0


def test_live_start_hook_focus_loss_never_reaches_sink(
    tmp_path: Path,
    supported_frame: np.ndarray,
) -> None:
    class BackgroundCapture(MockCapture):
        def is_foreground(self) -> bool:
            return False

    class PersistentReadyObserver:
        def observe(self, _frame, context):
            return PromptObservation(
                PromptObservationKind.READY_BITE,
                0.99,
                {PromptObservationKind.READY_BITE.value: 0.99},
                "stable_ready_test",
                context.frame_index,
                context.timestamp,
            )

    class NoInputSink:
        def __init__(self, _kwargs):
            self.calls = []

        def poll_panic(self):
            return False

        def apply(self, request, context):
            self.calls.append((request, context))
            raise AssertionError("focus loss must block START_HOOK")

        def summary(self):
            return {
                "action_sink_type": "sendinput",
                "action_allowlist": ["START_HOOK"],
                "panic_triggered": False,
            }

    created = []

    def factory(**kwargs):
        sink = NoInputSink(kwargs)
        created.append(sink)
        return sink

    capture = BackgroundCapture(supported_frame, diagnostics={
        "hwnd": 4242,
        "window_title": "test-window",
        "process": "BlackDesert64",
        "process_id": 99,
        "client_size": [2560, 1440],
    })
    runtime = _runtime(
        tmp_path,
        capture,
        FakeClock(),
        duration_seconds=0.8,
        emit_actions=True,
        action_sink_name="sendinput",
        action_allowlist="START_HOOK",
        action_sink_factory=factory,
    )
    runtime.prompt_bundle = replace(
        runtime.prompt_bundle,
        observer=PersistentReadyObserver(),
    )
    runtime.fsm.force_state(RuntimeState.WAITING, 0.0, "test_waiting")
    summary = runtime.run(max_frames=20)

    assert len(created) == 1
    assert created[0].calls == []
    assert summary["actions_applied"] == 0
    assert summary["unique_would_fire"].get("WOULD_START_HOOK", 0) == 0


def test_live_session_directory_never_overwrites(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 13, 1, 2, 3, tzinfo=timezone.utc)
    first = create_live_session_directory(tmp_path, timestamp=timestamp)
    second = create_live_session_directory(tmp_path, timestamp=timestamp)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_live_runtime_uses_no_third_party_input_and_never_loads_ground_truth() -> None:
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


def test_live_waiting_recovers_missed_ready_without_start_hook_action(
    tmp_path: Path,
    supported_frame: np.ndarray,
) -> None:
    class StableHookInstructionObserver:
        def observe(self, _frame, context):
            return PromptObservation(
                PromptObservationKind.HOOK_INSTRUCTION,
                0.97,
                {PromptObservationKind.HOOK_INSTRUCTION.value: 0.97},
                "session_20260723_150137",
                context.frame_index,
                context.timestamp,
                {},
            )

    runtime = _runtime(
        tmp_path,
        MockCapture(supported_frame),
        FakeClock(),
        duration_seconds=0.8,
    )
    runtime.prompt_bundle = replace(
        runtime.prompt_bundle,
        observer=StableHookInstructionObserver(),
    )
    runtime.fsm.force_state(RuntimeState.WAITING, 0.0, "test_waiting")
    summary = runtime.run(max_frames=30)

    events = [
        json.loads(line)
        for line in runtime.logger.events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    recovery = next(
        item
        for item in events
        if item["event_type"] == "missed_ready_recovered"
    )
    assert recovery["reason"] == "stable_hook_instruction_while_waiting"
    assert recovery["previous_state"] == "WAITING"
    assert recovery["next_state"] == "HOOK_PENDING"
    assert recovery["action_intent"] == ActionIntent.NONE.value
    assert recovery["action_applied"] is False
    assert summary["missed_ready_recovery_count"] == 1
    assert summary["unique_would_fire"].get("WOULD_START_HOOK", 0) == 0
    assert summary["raw_action_proposals"].get("START_HOOK", 0) == 0
    assert summary["detector_runs"]["hook"] > 0
    assert summary["actions_applied"] == 0


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

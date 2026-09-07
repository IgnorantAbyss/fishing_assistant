"""Exercise both real entrypoints/construction; never open capture or input."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_live_detect_only as live
from tools import run_fishing_production as production
from src.fishing_v2.live.live_detect_only import LiveDetectOnlyConfig, LiveDetectOnlyRuntime
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode


ACTION_ARGS = [
    "--enable-live-press-sequence", "--emit-actions", "true",
    "--action-sink", "sendinput", "--action-allowlist",
    "CAST,START_HOOK,HOOK_ACTION,PRESS_SEQUENCE,COLLECT",
]


def build_runtime(monkeypatch, tmp_path, entrypoint, extra=()):
    instances = []

    class InspectRuntime(LiveDetectOnlyRuntime):
        def run(self):
            # Real constructor/qualifier/scheduler, but no runtime loop, capture,
            # foreground API, sink initialization or SendInput.
            instances.append(self)
            assert self.action_sink is None
            return {"result": "completed", "actions_applied": 0}

    monkeypatch.setattr(live, "LiveDetectOnlyRuntime", InspectRuntime)
    monkeypatch.setattr(live, "resolve_window_target", lambda **kw: SimpleNamespace(
        window_title="test", window_info=None, title_prefix=None, resolution_mode="exact"))
    monkeypatch.setattr(live, "create_live_capture_session", lambda **kw: object())
    assert entrypoint.main([
        "--window-title", "test", "--runtime-profile", "production",
        "--output-dir", str(tmp_path), "--no-overlay", *ACTION_ARGS, *extra,
    ]) == 0
    return instances[0]


def semantics(runtime):
    config = runtime.live_config
    sink_factory = runtime.action_sink_factory
    sink_calls = []
    runtime.action_sink_factory = lambda **kw: sink_calls.append(kw) or object()
    runtime._capture_diagnostics = dict(
        hwnd=123, window_title="test", process="BlackDesert64",
        process_id=456, client_size=[2560, 1440])
    runtime._initialize_action_sink(session_started_at=0.0)
    assert len(sink_calls) == 1
    # Obtain actual evidence version from the selected adapter, not an assumed
    # mode-to-path mapping. Blank frame must still take the V3 temporal path.
    observation = runtime.press_detector.observe(
        np.zeros((1440, 2560, 3), dtype=np.uint8), FrameContext(1, 0.1))
    qualified, _ = runtime.controller.evidence_qualifier._press(
        observation, DetectorActivationMode.ACTIVE)
    return dict(
        mode=config.press_detector_mode, detector=type(runtime.press_detector),
        path=qualified.evidence["press_qualification_path"],
        qualifier=type(runtime.controller.evidence_qualifier),
        temporal=asdict(runtime.controller.evidence_qualifier.config),
        scheduler=type(runtime._press_live_emission),
        pacing=asdict(runtime._press_live_emission.config),
        enabled=runtime.enable_live_press_sequence,
        emit=runtime.emit_actions, sink=runtime.action_sink_name,
        sink_factory=sink_factory, allowlist=runtime.action_allowlist,
        sink_config=asdict(sink_calls[0]["config"]),
        sink_opt_in=sink_calls[0]["enable_live_press_sequence"],
        sink_allowlist=sink_calls[0]["allowlist"],
        anomaly=asdict(runtime._press_anomaly_evidence.config),
        result_anomaly=asdict(runtime._result_pending_timeout_evidence.config),
        profile=config.runtime_profile,
    )


@pytest.mark.parametrize("extra", [
    [],
    ["--press-anomaly-evidence"],
    ["--evidence-mode", "diagnostic", "--save-transition-frames", "--evidence-video-fps", "5"],
    ["--press-initial-delay-min-ms", "175", "--press-initial-delay-max-ms", "225",
     "--press-inter-key-gap-min-ms", "100", "--press-inter-key-gap-max-ms", "140",
     "--press-key-hold-ms", "45"],
])
def test_real_entrypoint_runtime_parity(monkeypatch, tmp_path, extra):
    a = semantics(build_runtime(monkeypatch, tmp_path / "live", live, extra))
    b = semantics(build_runtime(monkeypatch, tmp_path / "production", production, extra))
    assert a == b
    assert a["path"] == "temporal_v3"
    assert a["mode"] == "background-subtraction-live"
    if "--press-initial-delay-min-ms" not in extra:
        assert (a["pacing"]["initial_delay_min_ms"], a["pacing"]["initial_delay_max_ms"]) == (150, 250)
        assert (a["pacing"]["inter_key_gap_min_ms"], a["pacing"]["inter_key_gap_max_ms"]) == (90, 170)
        assert a["pacing"]["key_hold_ms"] == 40
    else:
        assert a["pacing"]["initial_delay_min_ms"] == 175
        assert a["pacing"]["initial_delay_max_ms"] == 225
        assert a["pacing"]["inter_key_gap_min_ms"] == 100
        assert a["pacing"]["inter_key_gap_max_ms"] == 140
        assert a["pacing"]["key_hold_ms"] == 45


def test_diagnostics_do_not_change_production_semantics(monkeypatch, tmp_path):
    plain = semantics(build_runtime(monkeypatch, tmp_path / "plain", live))
    diagnostic = semantics(build_runtime(monkeypatch, tmp_path / "diagnostic", live,
        ["--evidence-mode", "diagnostic", "--save-transition-frames", "--evidence-video-fps", "5"]))
    assert plain == diagnostic


def test_shared_profile_resolution_and_explicit_overrides():
    for profile in ("production", "diagnostic"):
        args = live.parse_args(["--window-title", "test", "--runtime-profile", profile])
        defaults = LiveDetectOnlyConfig.for_profile(profile)
        for name in ("press_initial_delay_min_ms", "press_initial_delay_max_ms",
                     "press_inter_key_gap_min_ms", "press_inter_key_gap_max_ms",
                     "press_key_hold_ms", "press_detector_mode"):
            assert getattr(args, name) == getattr(defaults, name)
        assert args.emit_actions is False and args.action_sink == "none"
        assert args.action_allowlist == "" and not args.enable_live_press_sequence
    args = live.parse_args(["--window-title", "test", "--press-detector-mode", "legacy"])
    assert args.press_detector_mode == "legacy"
    with pytest.raises(SystemExit):
        live.parse_args(["--window-title", "test", "--press-initial-delay-min-ms", "251"])

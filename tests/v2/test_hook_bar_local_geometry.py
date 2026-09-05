"""Bar-local ownership; real evidence is optional and never copied into Git."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from src.config_loader import load_roi_config
from src.detectors.hook_detector import detect_hook_bar
from src.fishing_v2.runtime.hook_crossing_geometry import measure_hook_crossing_geometry

ROOT = Path(__file__).resolve().parents[2]
ANOMALIES = ROOT / "reports/fishing_v2/production_v3/session_20260905_032141/hook_pending_anomalies"


def canvas(crop):
    image = np.zeros((1440, 2560, 3), np.uint8)
    x1, y1, x2, y2 = load_roi_config().pixel_roi("hook_bar_precise", 2560, 1440)
    image[y1:y2, x1:x2] = crop
    return image


@pytest.mark.parametrize("episode", [26, 27, 31, 34, 36])
def test_real_low_context_anomaly_has_one_bar_local_divider(episode):
    directory = ANOMALIES / f"episode_{episode}"
    if not directory.exists():
        pytest.skip("Local raw anomaly evidence is not distributed with source")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    samples = [s for s in manifest["samples"] if s["raw_observation"].get("fill_found")]
    assert samples
    for sample in samples:
        image = canvas(cv2.imread(str(directory / sample["raw_image_filename"])))
        result = detect_hook_bar(image, save_debug=False)
        assert result["structural_candidate"], (sample["frame_index"], result)
        geometry = measure_hook_crossing_geometry(image)
        assert geometry.divider_line_x == 1330
        assert geometry.fill_endpoint_x == sample["raw_observation"]["fill_endpoint_x"]


def test_hatched_components_do_not_split_real_fill_from_red_anchor(monkeypatch):
    import src.detectors.hook_detector as detector
    monkeypatch.setattr(detector, "_live_context_score", lambda *a, **kw: 0.30)
    crop = np.zeros((58, 614, 3), np.uint8)
    crop[16:41, 109:357] = (0, 0, 255)
    # Interior components must not replace the group's rightmost extent.
    for x in range(120, 220, 25):
        cv2.line(crop, (x, 20), (x+20, 37), (180, 140, 0), 1)
    crop[16:41, 357] = 255
    crop[19:39, 359:400] = (180, 140, 0)
    result = detect_hook_bar(canvas(crop), save_debug=False)
    assert result["structural_candidate"]


def test_divider_does_not_use_marker_or_diagonal_outside_bar():
    crop = np.zeros((58, 614, 3), np.uint8)
    crop[16:41, 109:357] = (0, 0, 255)
    crop[19:39, 359:400] = (180, 140, 0)
    cv2.line(crop, (335, 0), (375, 15), (255, 255, 255), 5)
    # Many white pixels outside the actual bar must not invent its divider.
    crop[:16, 357] = 255
    geometry = measure_hook_crossing_geometry(canvas(crop))
    assert not geometry.divider_line_detected


def structural_crop(endpoint=400):
    crop = np.zeros((58, 614, 3), np.uint8)
    crop[16:41, 109:357] = (0, 0, 255)
    crop[16:41, 357] = 255
    crop[19:39, 359:endpoint] = (180, 140, 0)
    return crop


def test_external_diagonal_pixels_cannot_change_fixed_anchor_measurement():
    crop = structural_crop()
    expected = measure_hook_crossing_geometry(canvas(crop))
    for offset in range(0, 15, 3):
        altered = crop.copy()
        cv2.line(altered, (320, offset), (400, 15), (255, 255, 255), 3)
        cv2.line(altered, (320, 42), (400, 57-offset), (255, 255, 255), 2)
        # Explicitly leave the current-frame anchor band untouched.
        altered[16:41] = crop[16:41]
        assert measure_hook_crossing_geometry(canvas(altered)) == expected


@pytest.mark.parametrize("fault", ["red_only", "no_divider", "weak_divider", "wrong_position", "no_fill", "no_shape", "conflict"])
def test_unsafe_structure_is_not_promoted(monkeypatch, fault):
    import src.detectors.hook_detector as detector
    monkeypatch.setattr(detector, "_live_context_score", lambda *a, **kw: 0.30)
    crop = structural_crop()
    if fault == "red_only":
        crop[:, 357:] = 0
    elif fault == "no_divider":
        crop[:, 357] = 0
    elif fault == "weak_divider":
        crop[24:41, 357] = 0  # Eight pixels cannot meet unchanged confidence.
    elif fault == "wrong_position":
        crop[:, 357] = 0
        crop[16:41, 410] = 255
    elif fault == "no_fill":
        crop[:, 359:] = 0
    elif fault == "no_shape":
        crop[16:35] = 0
    else:
        crop[16:41, 365] = 255
    result = detect_hook_bar(canvas(crop), save_debug=False)
    assert not result["structural_candidate"]
    assert not result["raw_detected"]


def test_production_adapter_reuses_single_geometry_pass(monkeypatch):
    import src.detectors.hook_detector as detector
    import src.fishing_v2.legacy_adapters.hook_detector_adapter as adapter_module
    from src.fishing_v2.domain.frame_context import FrameContext
    calls = []
    original = detector.measure_bar_local_geometry

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(detector, "measure_bar_local_geometry", measured)
    monkeypatch.setattr(adapter_module, "measure_hook_crossing_geometry", lambda *a: pytest.fail("second pass"))
    raw = adapter_module.LegacyHookDetectorAdapter().observe(canvas(structural_crop()), FrameContext(1, 0.025))
    assert calls == [1]
    assert raw.evidence["divider_line_x"] == 1330
    assert raw.evidence["fill_endpoint_x"] == 1372
    # No cached divider can survive a subsequent empty frame.
    missing = adapter_module.LegacyHookDetectorAdapter().observe(canvas(np.zeros((58,614,3), np.uint8)), FrameContext(2, 0.05))
    assert not missing.evidence["divider_line_detected"]


@pytest.mark.parametrize("foreground", [True, False])
def test_real_detector_common_qualification_crossing_and_safety(monkeypatch, foreground):
    import src.detectors.hook_detector as detector
    from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
    from src.fishing_v2.domain.frame_context import FrameContext
    from src.fishing_v2.domain.runtime_state import RuntimeState
    from src.fishing_v2.domain.action_intent import ActionIntent
    from src.fishing_v2.perception.observation_bundle import ObservationBundle
    from src.fishing_v2.fusion.observation_fusion import ObservationFusion
    from src.fishing_v2.runtime.fishing_fsm import FishingFSM, FSMConfig
    from src.fishing_v2.runtime.runtime_controller import RuntimeController, ActionExecutionMode
    from src.fishing_v2.runtime.safety_policy import SafetyPolicy, SafetyConfig
    monkeypatch.setattr(detector, "_live_context_score", lambda *a, **kw: 0.30)
    adapter = LegacyHookDetectorAdapter()
    controller = RuntimeController(ObservationFusion(), FishingFSM(FSMConfig(stable_frames=2), initial_state=RuntimeState.HOOK_PENDING), SafetyPolicy(SafetyConfig(emit_actions=False)))
    outcomes = []
    for i, endpoint in enumerate([367, 367, 400, 400, 400], 1):
        raw = adapter.observe(canvas(structural_crop(endpoint)), FrameContext(i, i*.025))
        outcomes.append(controller.process(ObservationBundle(i, i*.025, None, raw), foreground=foreground, runtime_environment_supported=True, action_mode=ActionExecutionMode.RECORDED_OBSERVATION, preserve_proposal=True))
    assert outcomes[0].fsm.next_state == RuntimeState.HOOK_PENDING
    assert outcomes[1].fsm.next_state == RuntimeState.HOOK
    assert [i for i, r in enumerate(outcomes, 1) if r.fsm.action_request.intent == ActionIntent.HOOK_ACTION] == [3]
    assert outcomes[2].safety.reason == ("action_emission_disabled" if foreground else "foreground_window_not_confirmed")


def test_anomaly_motion_is_passive_and_missing_geometry_breaks_history():
    from src.fishing_v2.live.hook_pending_timeout_evidence import HookPendingTimeoutEvidenceRecorder as Recorder
    def raw(endpoint, timestamp):
        return {"geometry_valid": True, "fill_endpoint_x": endpoint,
                "detector_frame_timestamp": timestamp, "divider_x": 1330,
                "bar_local_geometry": {"bar_inner_right": None}}
    before = raw(1369, 1.0)
    after = raw(1476, 1.025)
    motion = Recorder._motion_payload(after, before)
    assert motion["fill_direction"] == "expanding"
    assert motion["estimated_fill_velocity_px_per_sec"] == pytest.approx(4280)
    assert motion["distance_to_bar_right"] is None
    assert Recorder._motion_payload(raw(1469, 1.05), after)["fill_direction"] == "retracting"
    assert Recorder._motion_payload(after, {})["fill_direction"] == "unknown"
    assert Recorder._motion_payload(after, after)["estimated_fill_velocity_px_per_sec"] is None
    assert "motion" not in before and "motion" not in after

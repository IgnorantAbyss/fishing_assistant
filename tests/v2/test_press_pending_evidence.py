"""Synthetic boundary proof, NOT a replay of missing historical raw pixels."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from src.fishing_v2.live.press_anomaly_evidence import PressAnomalyEvidenceRecorder, PressAnomalyEvidenceConfig
from src.fishing_v2.runtime.press_sequence_aggregator import PressSequenceTemporalAggregator
from src.fishing_v2.runtime.runtime_controller import RuntimeController, ActionExecutionMode
from src.fishing_v2.runtime.fishing_fsm import FishingFSM
from src.fishing_v2.runtime.safety_policy import SafetyPolicy, SafetyConfig
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from .test_press_completeness import _press


def sample(frame, version=3):
    obs = _press(frame, "AADA")
    return replace(obs, evidence={**obs.evidence,
        "press_evidence_version": version, "input_effect_detected": True,
        "clean_frame_eligible": False, "frame_structurally_complete": True,
        "panel_phase": "PANEL_INPUT_STARTED"})


def test_v2_empty_clean_history_is_not_glyph_failure_and_v3_requires_consensus():
    old, current = PressSequenceTemporalAggregator(), PressSequenceTemporalAggregator()
    for frame in range(1, 6):
        v2 = old.update(sample(frame, 2))
        v3 = current.update(sample(frame, 3))
        assert v2.completeness.decoded_count == 4
        assert not v2.completeness.complete
        assert v2.completeness.occupancy_stability_count == 0
        assert not v2.completeness.panel_bbox_stable
        if frame == 1:
            assert not v3.sequence_ready  # Fully decoded is not sufficient.
    assert v3.sequence_ready and v3.sequence_candidate == tuple("AADA")


def test_unstable_geometry_still_abstains_v3():
    agg = PressSequenceTemporalAggregator()
    for frame in range(1, 4):
        obs = sample(frame)
        obs = replace(obs, evidence={**obs.evidence,
            "panel_bbox": [80, 280 + frame * 50, 500, 360 + frame * 50]})
        result = agg.update(obs)
        assert not result.sequence_ready


def record(rec, frame, timestamp):
    rec.record(episode_index=1, frame_index=frame, timestamp=timestamp,
        roi_pixels=np.zeros((202, 614, 3), np.uint8), observation=sample(frame),
        runtime_state="PRESS", certificate={"complete": False,
            "decoded_count": 4, "occupied_count": 4, "panel_bbox_stable": False})


@pytest.mark.parametrize("shutdown", [False, True])
def test_pending_or_immediate_shutdown_flushes_bounded_preroll(tmp_path, shutdown):
    rec = PressAnomalyEvidenceRecorder(tmp_path, PressAnomalyEvidenceConfig(
        enabled=True, frames_per_anomaly=12))
    for i in range(70):
        record(rec, i, i * .05)
    assert len(rec._samples) <= 12
    assert len(rec._pinned) == 1
    assert len(rec._pinned[1]) <= 8
    assert not rec.service_pending(episode_index=1, timestamp=0, runtime_state="PRESS",
        certificate={"complete": False}, authoritative_progress={})
    assert rec.service_pending(episode_index=1, timestamp=3.5, runtime_state="PRESS",
        certificate=None if shutdown else {"complete": False},
        authoritative_progress={}, shutdown=shutdown)
    assert not rec.service_pending(episode_index=1, timestamp=4, runtime_state="PRESS",
        certificate={"complete": False}, authoritative_progress={})
    assert rec.close()["press_anomaly_episode_count"] == 1
    folder = tmp_path / "press_anomalies/episode_1"
    manifest = json.loads((folder / "anomaly.json").read_text())
    assert manifest["reason"] == ("press_pending_at_shutdown" if shutdown else "press_completeness_prolonged_pending")
    assert 0 in manifest["frames"] and 69 in manifest["frames"]
    assert len(list(folder.glob("*_raw.png"))) <= 12


def test_success_and_short_pending_never_trigger(tmp_path):
    rec = PressAnomalyEvidenceRecorder(tmp_path, PressAnomalyEvidenceConfig(enabled=True))
    record(rec, 1, 0)
    for t, cert, progress in [(0, False, {}), (2, False, {}), (3, True, {}),
                              (8, False, {"sequence_frozen": True})]:
        assert not rec.service_pending(episode_index=1, timestamp=t, runtime_state="PRESS",
            certificate={"complete": cert}, authoritative_progress=progress, shutdown=True)
    rec.close()
    assert not rec.root.exists()


def test_record_failure_is_fail_open_and_memory_does_not_grow_across_episodes(tmp_path):
    rec = PressAnomalyEvidenceRecorder(tmp_path, PressAnomalyEvidenceConfig(enabled=True))
    for ep in range(1000):
        rec.record(episode_index=ep, frame_index=ep, timestamp=float(ep),
            roi_pixels=np.zeros((4, 4, 3), np.uint8), observation=sample(ep), certificate={})
    assert len(rec._pinned) == 1 and len(rec._samples) == 1
    rec.record(episode_index=1001, frame_index=1001, timestamp=1001,
        roi_pixels=np.zeros((1200,1200,3), np.uint8), observation=sample(1001), certificate={})
    assert not rec.enabled
    assert rec.close()["press_anomaly_evidence_failures"]


def test_successful_v3_controller_behavior_identical_with_evidence(tmp_path):
    outputs = []
    for enabled in (False, True):
        rec = PressAnomalyEvidenceRecorder(tmp_path / str(enabled),
            PressAnomalyEvidenceConfig(enabled=enabled))
        controller = RuntimeController(ObservationFusion(), FishingFSM(initial_state=RuntimeState.PRESS),
            SafetyPolicy(SafetyConfig(emit_actions=False)))
        rows = []
        for frame in range(1, 10):
            obs = sample(frame)
            result = controller.process(ObservationBundle(frame, obs.timestamp, press=obs),
                foreground=True, runtime_environment_supported=True,
                action_mode=ActionExecutionMode.RECORDED_OBSERVATION, preserve_proposal=True)
            rows.append(asdict(result))
            cert = result.qualified.bundle.press.evidence.get("press_completeness_certificate")
            rec.record(episode_index=1, frame_index=frame, timestamp=obs.timestamp,
                roi_pixels=np.zeros((202,614,3), np.uint8), observation=obs, certificate=cert)
            rec.service_pending(episode_index=1, timestamp=obs.timestamp,
                runtime_state=controller.fsm.state.value, certificate=cert,
                authoritative_progress={"sequence_frozen": bool(result.qualified.bundle.press.sequence_ready)})
        assert rec.close()["press_anomaly_episode_count"] == 0
        outputs.append(rows)
    assert outputs[0] == outputs[1]
    intents = [r["fsm"]["action_request"]["intent"].value for r in outputs[0]]
    assert intents.count("PRESS_SEQUENCE") == 1
    assert all(not row["action_applied"] for row in outputs[0])


@pytest.mark.parametrize("stop", ["panic", "manual"])
def test_real_runtime_finally_flushes_pending_without_capture_or_input(tmp_path, monkeypatch, stop):
    from types import SimpleNamespace
    from .test_live_detect_only import _runtime, MockCapture, FakeClock
    clock = FakeClock()
    capture = MockCapture(np.zeros((1440,2560,3), np.uint8))
    runtime = _runtime(tmp_path, capture, clock, runtime_profile="production",
                       press_anomaly_evidence=True, duration_seconds=10)
    runtime.fsm.force_state(RuntimeState.PRESS, 0, "synthetic_pending_fixture")
    from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
    runtime._press_shadow.observe(timestamp=0, frame_index=1,
        fsm_state=RuntimeState.PRESS, activation_mode=DetectorActivationMode.ACTIVE,
        raw=sample(1), qualified=None, qualification_reason="pending", safety_reason="disabled")
    rec = runtime._press_anomaly_evidence
    record(rec, 1, 0)
    rec.service_pending(episode_index=1, timestamp=0, runtime_state="PRESS",
                        certificate={"complete": False}, authoritative_progress={})
    def preflight():
        clock.value = 3.5
        if stop == "manual":
            raise KeyboardInterrupt()
    monkeypatch.setattr(runtime, "preflight", preflight)
    if stop == "panic":
        runtime.action_sink = SimpleNamespace(poll_panic=lambda: True)
    result = runtime.run()
    assert result["actions_applied"] == 0
    assert capture.calls == 0
    assert result["press_anomaly_episode_count"] == 1
    assert result["result"] == ("panic_shutdown" if stop == "panic" else "interrupted_by_user")

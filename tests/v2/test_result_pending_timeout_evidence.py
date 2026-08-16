from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.result_pending_timeout_evidence import (
    ResultPendingTimeoutEvidenceConfig,
    ResultPendingTimeoutEvidenceRecorder,
)
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import EvidenceQualification


def _raw_press(frame: int, *, panel: bool = True) -> PressObservation:
    slots = [{
        "index": 0,
        "occupancy": "UNCERTAIN",
        "arrow_candidate": "DOWN",
        "arrow_confidence": 0.41,
        "letter_candidate": "S",
        "letter_confidence": 0.77,
        "mapped_key": None,
        "confidence": 0.0,
    }]
    return PressObservation(
        detected=panel,
        confidence=0.91 if panel else 0.0,
        frame_index=frame,
        timestamp=frame / 10.0,
        panel_candidate=panel,
        panel_present=panel,
        sequence_candidate=(),
        sequence_confidence=0.0,
        evidence={
            "press_evidence_version": 3,
            "panel_phase": "PANEL_APPEARING",
            "slots": slots,
            "v3": {
                "panel_present": panel,
                "geometry_stable": panel,
                "key_strip_bbox": [10, 20, 110, 50] if panel else None,
                "sequence_candidate": [],
                "occupied_slot_count": 1 if panel else 0,
                "frame_complete": False,
                "sequence_confidence": 0.0,
                "slots": slots,
                "input_effect_detected": False,
                "episode_input_started": False,
                "post_input_frame": False,
                "panel_phase": "PANEL_APPEARING",
            },
        },
    )


def _qualification() -> EvidenceQualification:
    return EvidenceQualification(
        "press",
        DetectorActivationMode.ARMED,
        True,
        False,
        "clean_sequence_not_ready",
        False,
        False,
        sequence_ready=False,
        sequence_qualification_reason="sequence_consensus_pending",
    )


def _begin(recorder: ResultPendingTimeoutEvidenceRecorder) -> None:
    assert recorder.begin(
        runtime_state=RuntimeState.RESULT_PENDING,
        started_at=10.0,
        cycle_id=7,
        hook_action_identity="cycle:7:HOOK_ACTION",
    )


def _record(
    recorder: ResultPendingTimeoutEvidenceRecorder,
    *,
    frame: int,
    timestamp: float,
    detector_executed: bool = True,
    panel: bool = True,
) -> bool:
    raw = _raw_press(frame, panel=panel) if detector_executed else None
    return recorder.record(
        frame_index=frame,
        timestamp=timestamp,
        runtime_state=RuntimeState.RESULT_PENDING,
        roi_pixels=np.full((24, 40, 3), frame % 255, dtype=np.uint8),
        press_detector_executed=detector_executed,
        detector_mode=DetectorActivationMode.ARMED,
        cadence={"detector_due": detector_executed, "target_fps": 5.0},
        raw_observation=raw,
        qualified_observation=raw,
        qualification=_qualification() if detector_executed else None,
        get_evidence_seen=False,
    )


def test_buffer_only_starts_in_result_pending(tmp_path: Path) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(
        tmp_path,
        ResultPendingTimeoutEvidenceConfig(enabled=True),
    )

    assert not recorder.begin(
        runtime_state=RuntimeState.HOOK,
        started_at=1.0,
        cycle_id=1,
        hook_action_identity="cycle:1:HOOK_ACTION",
    )
    assert not recorder.active
    assert recorder.close()["result_pending_timeout_evidence_saved_count"] == 0


def test_sampling_is_cadence_bounded_and_buffer_is_capped(tmp_path: Path) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(
        tmp_path,
        ResultPendingTimeoutEvidenceConfig(
            enabled=True,
            sample_interval_seconds=0.2,
            max_samples=60,
        ),
    )
    _begin(recorder)
    accepted = sum(
        _record(recorder, frame=index, timestamp=10.0 + index * 0.01)
        for index in range(1000)
    )

    assert 49 <= accepted <= 51
    assert recorder.sample_count <= 60
    recorder.finish(
        next_state=RuntimeState.SYNC_REQUIRED,
        timestamp=20.1,
        reason="result_pending_maximum_timeout",
    )
    summary = recorder.close()
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    assert summary["result_pending_timeout_evidence_saved_count"] == 1
    assert manifest["sample_count"] == accepted


@pytest.mark.parametrize(
    ("next_state", "reason"),
    [
        (RuntimeState.PRESS, "stable_highest_supported_candidate"),
        (RuntimeState.GET, "get_panel_priority"),
        (RuntimeState.IDLE, "authoritative_idle_prompt_recovery"),
    ],
)
def test_normal_result_resolution_discards_without_disk_write(
    tmp_path: Path,
    next_state: RuntimeState,
    reason: str,
) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(
        tmp_path,
        ResultPendingTimeoutEvidenceConfig(enabled=True),
    )
    _begin(recorder)
    _record(recorder, frame=1, timestamp=10.0)
    assert not recorder.finish(
        next_state=next_state,
        timestamp=10.5,
        reason=reason,
    )
    assert recorder.close()["result_pending_timeout_evidence_saved_count"] == 0
    assert not (tmp_path / "result_pending_anomalies").exists()


def test_timeout_preserves_raw_roi_when_panel_locator_misses(tmp_path: Path) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(
        tmp_path,
        ResultPendingTimeoutEvidenceConfig(enabled=True),
    )
    _begin(recorder)
    _record(recorder, frame=7, timestamp=10.0, panel=False)
    assert recorder.finish(
        next_state=RuntimeState.SYNC_REQUIRED,
        timestamp=20.0,
        reason="result_pending_maximum_timeout",
    )
    summary = recorder.close()
    manifest_path = next(tmp_path.rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = manifest["samples"][0]

    assert summary["result_pending_timeout_evidence_saved_count"] == 1
    assert sample["raw_observation"]["panel_found"] is False
    assert (manifest_path.parent / sample["raw_image_filename"]).is_file()


def test_not_executed_and_incomplete_candidates_remain_distinguishable(
    tmp_path: Path,
) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(
        tmp_path,
        ResultPendingTimeoutEvidenceConfig(enabled=True),
    )
    _begin(recorder)
    _record(
        recorder,
        frame=1,
        timestamp=10.0,
        detector_executed=False,
    )
    _record(recorder, frame=2, timestamp=10.3, panel=True)
    recorder.finish(
        next_state=RuntimeState.SYNC_REQUIRED,
        timestamp=20.0,
        reason="result_pending_maximum_timeout",
    )
    recorder.close()
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())

    assert manifest["samples"][0]["press_detector_executed"] is False
    assert manifest["samples"][0]["raw_observation"]["status"] == (
        "not_evaluated"
    )
    assert manifest["samples"][0]["qualification"]["status"] == (
        "not_evaluated"
    )
    assert manifest["samples"][1]["raw_observation"]["frame_complete"] is False
    assert manifest["samples"][1]["raw_observation"]["slots"][0][
        "occupancy"
    ] == "UNCERTAIN"


def test_repeated_timeouts_obey_episode_cap_and_io_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(
        tmp_path,
        ResultPendingTimeoutEvidenceConfig(enabled=True, max_episodes=2),
    )
    original = recorder._write_episode
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(recorder, "_write_episode", fail_first)
    for index in range(3):
        _begin(recorder)
        _record(recorder, frame=index + 1, timestamp=10.0)
        recorder.finish(
            next_state=RuntimeState.SYNC_REQUIRED,
            timestamp=20.0,
            reason="result_pending_maximum_timeout",
        )

    summary = recorder.close()
    assert summary["result_pending_timeout_count"] == 3
    assert summary["result_pending_timeout_evidence_saved_count"] == 1
    assert summary["result_pending_timeout_evidence_failure_count"] == 1
    assert summary["result_pending_timeout_evidence_cap_drop_count"] == 1


def test_disabled_recorder_does_not_copy_or_write(tmp_path: Path) -> None:
    recorder = ResultPendingTimeoutEvidenceRecorder(tmp_path)
    assert not recorder.begin(
        runtime_state=RuntimeState.RESULT_PENDING,
        started_at=1.0,
        cycle_id=1,
        hook_action_identity="cycle:1:HOOK_ACTION",
    )
    assert recorder.close()["result_pending_timeout_evidence_saved_count"] == 0
    assert list(tmp_path.iterdir()) == []

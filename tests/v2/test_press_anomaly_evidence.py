from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.live.press_anomaly_evidence import (
    PressAnomalyEvidenceConfig,
    PressAnomalyEvidenceRecorder,
)


def _observation(frame: int) -> PressObservation:
    return PressObservation(
        True,
        0.95,
        frame,
        frame / 20.0,
        panel_candidate=True,
        panel_present=True,
        sequence_candidate=("D",),
        sequence_confidence=0.9931,
        evidence={
            "panel_phase": "PANEL_CLEAN",
            "slots": [{
                "index": 0,
                "bbox": [2, 2, 20, 18],
                "arrow_bbox": [7, 9, 14, 16],
                "occupancy": "OCCUPIED",
                "mapped_key": "D",
                "arrow_confidence": 0.9931,
            }],
            "v3": {
                "geometry_continuity": {
                    "decision": "anchor_preserved_after_grid_phase_jump",
                    "origin_delta_pitch_fraction": 0.5,
                },
            },
        },
    )


def _record(
    recorder: PressAnomalyEvidenceRecorder,
    count: int = 12,
    *,
    episode_index: int = 1,
) -> None:
    for frame in range(1, count + 1):
        recorder.record(
            episode_index=episode_index,
            frame_index=frame,
            timestamp=frame / 20.0,
            roi_pixels=np.zeros((24, 40, 3), dtype=np.uint8),
            observation=_observation(frame),
            certificate={"complete": False, "occupied_count": 7, "decoded_count": 1},
        )


def test_genuine_pre_freeze_incomplete_episode_still_writes_evidence(
    tmp_path: Path,
) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder)

    assert recorder.incomplete_episode_eligible({}) is True
    assert recorder.trigger_incomplete(
        episode_index=1,
        reason="press_sequence_abstained_incomplete",
        authoritative_progress={},
    ) is True
    summary = recorder.close()

    assert summary["press_anomaly_episode_count"] == 1
    episode = tmp_path / "press_anomalies" / "episode_1"
    assert episode.is_dir()
    sample = json.loads(next(episode.glob("frame_*.json")).read_text("utf-8"))
    assert sample["geometry_continuity"]["decision"] == (
        "anchor_preserved_after_grid_phase_jump"
    )


@pytest.mark.parametrize(
    "progress",
    [
        {"sequence_frozen": True},
        {"opportunity_created": True},
        {"opportunity_scheduled": True},
        {
            "sequence_frozen": True,
            "opportunity_created": True,
            "opportunity_scheduled": True,
        },
        {"emission_started": True},
        {"action_applied": True},
    ],
)
def test_authoritative_press_progress_suppresses_incomplete_classification(
    tmp_path: Path,
    progress: dict[str, bool],
) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder)

    assert recorder.incomplete_episode_eligible(progress) is False
    assert recorder.trigger_incomplete(
        episode_index=1,
        reason="press_sequence_abstained_incomplete",
        authoritative_progress=progress,
    ) is False
    summary = recorder.close()

    assert summary["press_anomaly_episode_count"] == 0
    assert not (tmp_path / "press_anomalies").exists()


def test_completed_episode_does_not_hide_next_pre_freeze_failure(
    tmp_path: Path,
) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder, episode_index=1)
    _record(recorder, episode_index=2)

    assert recorder.trigger_incomplete(
        episode_index=1,
        reason="press_sequence_abstained_incomplete",
        authoritative_progress={
            "sequence_frozen": True,
            "opportunity_created": True,
            "opportunity_scheduled": True,
            "emission_started": True,
            "action_applied": True,
        },
    ) is False
    assert recorder.trigger_incomplete(
        episode_index=2,
        reason="press_sequence_abstained_incomplete",
        authoritative_progress={},
    ) is True
    summary = recorder.close()

    assert summary["press_anomaly_episode_count"] == 1
    assert not (tmp_path / "press_anomalies" / "episode_1").exists()
    assert (tmp_path / "press_anomalies" / "episode_2").is_dir()


def test_action_applied_before_visual_ack_suppresses_disappearance_anomaly(
    tmp_path: Path,
) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder)

    assert recorder.trigger_incomplete(
        episode_index=1,
        reason="press_sequence_abstained_incomplete",
        authoritative_progress={
            "sequence_frozen": True,
            "opportunity_created": True,
            "opportunity_scheduled": True,
            "emission_started": True,
            "action_applied": True,
        },
    ) is False
    summary = recorder.close()

    assert summary["press_anomaly_episode_count"] == 0
    assert not (tmp_path / "press_anomalies").exists()


def test_anomaly_io_failure_does_not_escape_control_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder)

    def fail_writer(*_args) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_write_episode", fail_writer)
    assert recorder.trigger_incomplete(
        episode_index=1,
        reason="press_sequence_abstained_incomplete",
        authoritative_progress={},
    ) is True
    summary = recorder.close()

    assert summary["press_anomaly_episode_count"] == 1
    assert summary["press_anomaly_evidence_failures"] == [
        "OSError: disk full"
    ]


def test_disabled_anomaly_evidence_writes_nothing(tmp_path: Path) -> None:
    recorder = PressAnomalyEvidenceRecorder(tmp_path)
    _record(recorder)
    assert recorder.trigger(episode_index=1, reason="timeout") is False
    summary = recorder.close()
    assert summary["press_anomaly_evidence_path"] is None
    assert list(tmp_path.rglob("*.png")) == []


def test_successful_episode_does_not_create_anomaly_files(tmp_path: Path) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder)
    summary = recorder.close()
    assert summary["press_anomaly_episode_count"] == 0
    assert not (tmp_path / "press_anomalies").exists()


def test_timeout_writes_only_six_bounded_roi_samples_and_no_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_writer_calls: list[object] = []
    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        lambda *args, **kwargs: video_writer_calls.append((args, kwargs)),
    )
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(
            enabled=True,
            buffer_frames=12,
            max_episodes=20,
            frames_per_anomaly=6,
        ),
    )
    _record(recorder, 20)
    assert recorder.trigger(
        episode_index=1,
        reason="press_visual_ack_timeout",
    ) is True
    summary = recorder.close()
    episode = tmp_path / "press_anomalies" / "episode_1"
    assert summary["press_anomaly_episode_count"] == 1
    assert len(list(episode.glob("*_raw.png"))) == 6
    assert len(list(episode.glob("*_occupied.png"))) == 6
    assert len(list(episode.glob("*_arrows.png"))) == 6
    assert len(list(episode.glob("frame_*.json"))) == 6
    assert list(episode.glob("*.mp4")) == []
    assert video_writer_calls == []


def test_anomaly_disk_writer_runs_after_nonblocking_trigger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True),
    )
    _record(recorder)
    writer_started = threading.Event()
    allow_writer_to_finish = threading.Event()

    def slow_writer(*_args) -> None:
        writer_started.set()
        assert allow_writer_to_finish.wait(timeout=2.0)

    monkeypatch.setattr(recorder, "_write_episode", slow_writer)
    assert recorder.trigger(episode_index=1, reason="timeout") is True
    assert writer_started.wait(timeout=1.0)
    # The capture/runtime caller already regained control while the writer waits.
    allow_writer_to_finish.set()
    summary = recorder.close()
    assert summary["press_anomaly_evidence_failures"] == []


def test_anomaly_preserves_early_best_and_mismatch_categories(tmp_path: Path) -> None:
    recorder = PressAnomalyEvidenceRecorder(
        tmp_path,
        PressAnomalyEvidenceConfig(enabled=True, buffer_frames=12, frames_per_anomaly=6),
    )
    for frame in range(1, 21):
        decoded = 4 if frame == 3 else 0
        observation = _observation(frame)
        if decoded == 4:
            observation = replace(
                observation,
                sequence_candidate=tuple("WWAW"),
            )
        recorder.record(
            episode_index=1,
            frame_index=frame,
            timestamp=frame / 20.0,
            roi_pixels=np.zeros((24, 40, 3), dtype=np.uint8),
            observation=observation,
            certificate={
                "complete": False,
                "occupied_count": 10 if frame <= 3 else 4,
                "decoded_count": decoded,
                "sequence_stability_count": 3 if frame == 3 else 0,
            },
        )
    assert recorder.trigger(episode_index=1, reason="timeout") is True
    recorder.close()
    anomaly = json.loads(
        (tmp_path / "press_anomalies" / "episode_1" / "anomaly.json").read_text(
            encoding="utf-8"
        )
    )
    assert anomaly["evidence_categories"]["best_decoded_candidate"] == 3
    assert anomaly["evidence_categories"]["first_incomplete_stable_candidate"] == 3
    assert anomaly["evidence_categories"]["first_occupancy_decoded_mismatch"] == 1
    assert anomaly["evidence_categories"]["last_before_panel_disappears"] == 20
    assert 3 in anomaly["frames"]

from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

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
        },
    )


def _record(recorder: PressAnomalyEvidenceRecorder, count: int = 12) -> None:
    for frame in range(1, count + 1):
        recorder.record(
            episode_index=1,
            frame_index=frame,
            timestamp=frame / 20.0,
            roi_pixels=np.zeros((24, 40, 3), dtype=np.uint8),
            observation=_observation(frame),
            certificate={"complete": False, "occupied_count": 7, "decoded_count": 1},
        )


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

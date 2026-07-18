from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.fishing_v2.live.diagnostic_evidence import (
    DiagnosticEvidenceConfig,
    DiagnosticEvidenceRecorder,
)


class FakeVideoWriter:
    def __init__(self, path: Path, *, opened: bool) -> None:
        self.path = path
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False
        if opened:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class EvidenceIO:
    def __init__(self) -> None:
        self.writers: list[tuple[str, float, tuple[int, int], FakeVideoWriter]] = []
        self.roi_shapes: dict[str, tuple[int, ...]] = {}

    def video_writer(
        self, path: Path, codec: str, fps: float, size: tuple[int, int]
    ) -> FakeVideoWriter:
        writer = FakeVideoWriter(path, opened=codec == "mp4v")
        self.writers.append((codec, fps, size, writer))
        return writer

    def image_writer(self, path: Path, crop: np.ndarray, _quality: int) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"roi")
        self.roi_shapes[path.parent.name] = crop.shape
        return True

    @property
    def active_writer(self) -> FakeVideoWriter:
        return next(item[3] for item in self.writers if item[3].opened)


def _recorder(tmp_path: Path, io: EvidenceIO, *, fps: float = 10.0) -> DiagnosticEvidenceRecorder:
    return DiagnosticEvidenceRecorder(
        tmp_path / "session",
        config=DiagnosticEvidenceConfig(video_fps=fps),
        writer_factory=io.video_writer,
        image_writer=io.image_writer,
    )


def test_full_session_video_uses_original_shape_and_timestamp_sidecar(tmp_path: Path) -> None:
    io = EvidenceIO()
    recorder = _recorder(tmp_path, io)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 1] = 77
    assert recorder.record_frame(frame, capture_frame_index=11, timestamp=0.0)
    assert not recorder.record_frame(frame, capture_frame_index=12, timestamp=0.04)
    assert recorder.record_frame(frame, capture_frame_index=13, timestamp=0.10)
    summary = recorder.finalize()

    assert [item[0] for item in io.writers] == ["avc1", "mp4v"]
    assert io.writers[-1][1:3] == (10.0, (6, 4))
    assert [item.shape for item in io.active_writer.frames] == [(4, 6, 3), (4, 6, 3)]
    assert np.all(io.active_writer.frames[0][:, :, 1] == 77)
    assert io.active_writer.released is True
    assert summary["video_codec"] == "mp4v"
    assert summary["requested_video_codec"] == "avc1"
    assert summary["attempted_codecs"] == ["avc1", "mp4v"]
    assert summary["actual_video_codec"] == "mp4v"
    assert summary["video_codec_fallback_used"] is True
    assert summary["codec_initialization_errors"] == [
        {"codec": "avc1", "reason": "writer_is_opened_false"}
    ]
    assert Path(summary["actual_video_path"]).name == "session_capture_mp4v.mp4"
    assert summary["video_frame_size"] == [6, 4]
    assert summary["video_frame_count"] == 2
    assert summary["first_timestamp"] == 0.0
    assert summary["last_timestamp"] == 0.1
    rows = list(csv.DictReader(Path(summary["video_index_path"]).open(encoding="utf-8")))
    assert [(row["capture_frame_index"], row["timestamp"]) for row in rows] == [
        ("11", "0.000000000"), ("13", "0.100000000"),
    ]


def test_codec_factory_exception_is_recorded_before_mp4v_fallback(tmp_path: Path) -> None:
    io = EvidenceIO()

    def writer_factory(path, codec, fps, size):
        if codec == "avc1":
            raise RuntimeError("simulated incompatible OpenH264")
        return io.video_writer(path, codec, fps, size)

    recorder = DiagnosticEvidenceRecorder(
        tmp_path / "session",
        writer_factory=writer_factory,
        image_writer=io.image_writer,
    )
    assert recorder.record_frame(
        np.zeros((4, 6, 3), dtype=np.uint8),
        capture_frame_index=1,
        timestamp=0.0,
    )
    summary = recorder.finalize()
    assert summary["actual_video_codec"] == "mp4v"
    assert summary["video_codec_fallback_used"] is True
    assert summary["codec_initialization_errors"] == [{
        "codec": "avc1",
        "reason": "RuntimeError: simulated incompatible OpenH264",
    }]


def test_both_video_codecs_failing_is_explicit(tmp_path: Path) -> None:
    def closed_writer(path, codec, _fps, _size):
        return FakeVideoWriter(path, opened=False)

    recorder = DiagnosticEvidenceRecorder(
        tmp_path / "session", writer_factory=closed_writer
    )
    with pytest.raises(RuntimeError, match="avc1.*mp4v"):
        recorder.record_frame(
            np.zeros((4, 6, 3), dtype=np.uint8),
            capture_frame_index=1,
            timestamp=0.0,
        )
    summary = recorder.finalize()
    assert summary["attempted_codecs"] == ["avc1", "mp4v"]
    assert summary["actual_video_codec"] is None
    assert summary["actual_video_path"] is None
    assert len(summary["codec_initialization_errors"]) == 2


def test_real_mp4v_fallback_is_readable_at_live_resolution(tmp_path: Path) -> None:
    def force_mp4v(path, codec, fps, size):
        if codec == "avc1":
            return FakeVideoWriter(path, opened=False)
        return cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size, True
        )

    recorder = DiagnosticEvidenceRecorder(
        tmp_path / "session", writer_factory=force_mp4v
    )
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    for index in range(3):
        frame[:, :, 1] = index * 40
        assert recorder.record_frame(
            frame, capture_frame_index=index + 1, timestamp=index / 10.0
        )
    summary = recorder.finalize()

    path = Path(summary["actual_video_path"])
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2560
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1440
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        readable, decoded = capture.read()
        assert readable is True
        assert decoded.shape == (1440, 2560, 3)
    finally:
        capture.release()
    assert summary["actual_video_codec"] == "mp4v"
    assert summary["dropped_video_frames"] == 0


def test_dense_roi_evidence_exists_without_would_fire_event(tmp_path: Path) -> None:
    io = EvidenceIO()
    recorder = _recorder(tmp_path, io)
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    recorder.record_frame(frame, capture_frame_index=1, timestamp=0.0)
    bounds = {
        "prompt": (0, 0, 10, 4),
        "hook": (5, 4, 20, 10),
        "press": (10, 8, 30, 16),
        "get": (25, 5, 40, 20),
    }
    recorder.record_detector_evidence(
        frame,
        capture_frame_index=1,
        timestamp=0.0,
        episode_id=2,
        runtime_state="WAITING",
        roi_bounds=bounds,
        detector_metadata={
            "prompt": {"raw": "UNKNOWN", "qualified": False, "rejection_reason": "low_similarity"},
            "hook": {"raw": False, "qualified": False, "rejection_reason": "raw_not_detected"},
            "press": {"raw": False, "qualified": False, "rejection_reason": "raw_not_detected"},
            "get": {"raw": False, "qualified": False, "rejection_reason": "raw_not_detected"},
        },
        executed={"prompt": True, "hook": True, "press": False, "get": False},
    )
    summary = recorder.finalize()

    assert summary["video_frame_count"] == 1
    assert summary["roi_evidence_counts_by_episode"]["2"] == {
        "prompt": 1, "hook": 1, "press": 1, "get": 1,
    }
    assert summary["detector_execution_counts_by_episode"]["2"] == {
        "prompt": 1, "hook": 1, "press": 0, "get": 0,
    }
    assert io.roi_shapes == {
        "prompt": (4, 10, 3),
        "hook": (6, 15, 3),
        "press": (8, 20, 3),
        "get": (15, 15, 3),
    }
    records = [json.loads(line) for line in Path(summary["detector_evidence_path"]).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["runtime_state"] == "WAITING"
    assert records[0]["detectors"]["prompt"]["rejection_reason"] == "low_similarity"
    assert set(records[0]["roi_paths"]) == {"prompt", "hook", "press", "get"}


def test_get_diagnostic_metadata_keeps_complete_legacy_debug() -> None:
    from src.fishing_v2.domain.observations import GetObservation
    from src.fishing_v2.live.live_detect_only import LiveDetectOnlyRuntime
    from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
    from src.fishing_v2.runtime.detector_evidence import EvidenceQualification

    debug = {
        "search_roi": [1, 2, 3, 4],
        "panel_bbox": [10, 20, 110, 220],
        "panel_bbox_global": [11, 22, 113, 224],
        "localization_source": "vertical_sliding_strong_grid",
        "panel_confidence": 0.96,
        "structure_debug": {
            "grid_cell_candidates": 12,
            "title_bright_ratio": 0.03,
            "button_bright_ratio": 0.02,
        },
        "vertical_sliding": {
            "selected": {
                "vertical_anchor_px": 42,
                "vertical_offset_ratio": 0.1,
                "dark_ratio": 0.91,
            }
        },
    }
    raw = GetObservation(True, 0.96, 7, 1.4, evidence={
        "matched_features": ["inventory_title", "item_grid", "collect_button"],
        "legacy_debug": debug,
    })
    qualified = GetObservation(True, 0.96, 7, 1.4, evidence={
        **raw.evidence, "get_confirmation_frames": 2,
    })
    qualification = EvidenceQualification(
        "get", DetectorActivationMode.BURST, True, True,
        "strong_get_panel_temporally_qualified", True, False,
    )
    summary = LiveDetectOnlyRuntime._observation_summary(raw)
    fields = LiveDetectOnlyRuntime._get_diagnostic_fields(raw, qualified, qualification)
    assert summary["evidence"]["legacy_debug"] == debug
    assert fields == {
        "raw_candidate": True,
        "qualified": True,
        "confidence": 0.96,
        "search_roi": [1, 2, 3, 4],
        "candidate_bbox": [10, 20, 110, 220],
        "candidate_bbox_global": [11, 22, 113, 224],
        "vertical_anchor_px": 42,
        "vertical_offset_ratio": 0.1,
        "panel_confidence": 0.96,
        "fallback_used": True,
        "fallback_reason": None,
        "localization_source": "vertical_sliding_strong_grid",
        "dark_ratio": 0.91,
        "grid_contour_count": 12,
        "title_bright_ratio": 0.03,
        "button_bright_ratio": 0.02,
        "temporal_confirmation_count": 2,
        "activation_mode": "BURST",
        "evidence_eligible_for_fusion": True,
        "rejection_reason": "strong_get_panel_temporally_qualified",
    }


def test_event_ring_indexes_two_seconds_before_and_after_in_full_video(tmp_path: Path) -> None:
    io = EvidenceIO()
    recorder = _recorder(tmp_path, io)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for index in range(41):
        timestamp = index / 10.0
        recorder.record_frame(frame, capture_frame_index=index + 1, timestamp=timestamp)
        if index == 20:
            recorder.mark_event("detector_conflict", {
                "timestamp": timestamp,
                "frame_index": index + 1,
            })
    summary = recorder.finalize()
    windows = [json.loads(line) for line in Path(summary["event_windows_path"]).read_text(encoding="utf-8").splitlines()]
    event = next(item for item in windows if item["event_type"] == "detector_conflict")
    assert event["first_capture_frame_index"] == 1
    assert event["last_capture_frame_index"] == 41
    assert event["sample_count"] == 41
    assert event["post_window_complete"] is True
    assert event["source"] == "full_session_video_ring_index"


def test_ctrl_c_style_early_finalize_releases_video_and_marks_partial_event_window(
    tmp_path: Path,
) -> None:
    io = EvidenceIO()
    recorder = _recorder(tmp_path, io)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder.record_frame(frame, capture_frame_index=1, timestamp=0.0)
    recorder.mark_event("prompt_label_change", {"timestamp": 0.0, "frame_index": 1})
    summary = recorder.finalize()
    event = json.loads(Path(summary["event_windows_path"]).read_text(encoding="utf-8"))
    assert io.active_writer.released is True
    assert event["post_window_complete"] is False
    assert summary["video_frame_count"] == 1


def test_dropped_video_slots_are_reported_as_evidence_gaps(tmp_path: Path) -> None:
    io = EvidenceIO()
    recorder = _recorder(tmp_path, io)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder.record_frame(frame, capture_frame_index=1, timestamp=0.0)
    recorder.record_frame(frame, capture_frame_index=2, timestamp=0.35)
    summary = recorder.finalize()
    assert summary["dropped_video_frames"] == 2
    assert summary["has_evidence_gaps"] is True
    assert summary["evidence_gap_intervals"][0]["missing_video_frames"] == 2

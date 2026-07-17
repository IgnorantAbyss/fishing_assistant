"""Bounded full-session video and dense ROI evidence for live diagnostics."""

from __future__ import annotations

from collections import defaultdict, deque
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from src.screen_capture import validate_bgr_frame


EVIDENCE_MODES = ("minimal", "diagnostic")
ROI_NAMES = ("prompt", "hook", "press", "get")


@dataclass(frozen=True)
class DiagnosticEvidenceConfig:
    video_fps: float = 10.0
    event_pre_seconds: float = 2.0
    event_post_seconds: float = 2.0
    jpeg_quality: int = 82

    def __post_init__(self) -> None:
        if self.video_fps <= 0:
            raise ValueError("diagnostic video_fps must be positive")
        if self.event_pre_seconds < 0 or self.event_post_seconds < 0:
            raise ValueError("diagnostic event window seconds must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("diagnostic jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class VideoSample:
    video_frame_index: int
    capture_frame_index: int
    timestamp: float


@dataclass
class PendingEventWindow:
    event_type: str
    event_timestamp: float
    event_frame_index: int
    samples: list[VideoSample]


def _open_video_writer(
    path: Path, codec: str, fps: float, size: tuple[int, int]
) -> Any:
    return cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, size, True
    )


def _write_jpeg(path: Path, crop: np.ndarray, quality: int) -> bool:
    return bool(cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, quality]))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _json_safe(enum_value)
    return str(value)


class DiagnosticEvidenceRecorder:
    """Write evidence only in explicit diagnostic mode.

    Event windows reference frames in the full-session video instead of
    duplicating full-resolution JPEGs. The in-memory ring contains only sample
    metadata; video pixels are already durably owned by the session writer.
    """

    def __init__(
        self,
        session_path: str | Path,
        *,
        config: DiagnosticEvidenceConfig | None = None,
        writer_factory: Callable[[Path, str, float, tuple[int, int]], Any] = _open_video_writer,
        image_writer: Callable[[Path, np.ndarray, int], bool] = _write_jpeg,
    ) -> None:
        self.session_path = Path(session_path)
        self.config = config or DiagnosticEvidenceConfig()
        self.root = self.session_path / "diagnostic_evidence"
        self.roi_root = self.root / "rois"
        self.video_index_path = self.root / "video_frames.csv"
        self.metadata_path = self.root / "detector_evidence.jsonl"
        self.event_windows_path = self.root / "event_windows.jsonl"
        self._writer_factory = writer_factory
        self._image_writer = image_writer
        self._writer: Any | None = None
        self._video_path: Path | None = None
        self._video_codec: str | None = None
        self._frame_size: tuple[int, int] | None = None
        self._next_due: float | None = None
        self._video_samples: list[VideoSample] = []
        self._ring: deque[VideoSample] = deque()
        self._pending_events: list[PendingEventWindow] = []
        self._completed_events: list[dict[str, Any]] = []
        self._dropped_video_frames = 0
        self._video_gaps: list[dict[str, Any]] = []
        self._last_capture_timestamp: float | None = None
        self._roi_counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {name: 0 for name in ROI_NAMES}
        )
        self._execution_counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {name: 0 for name in ROI_NAMES}
        )
        self._evidence_records = 0
        self._finalized = False

        self.root.mkdir(parents=True, exist_ok=True)
        for name in ROI_NAMES:
            (self.roi_root / name).mkdir(parents=True, exist_ok=True)
        with self.video_index_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=(
                "video_frame_index", "capture_frame_index", "timestamp",
            )).writeheader()

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        candidates = (
            ("avc1", self.root / "session_capture_h264.mp4"),
            ("mp4v", self.root / "session_capture.mp4"),
            ("MJPG", self.root / "session_capture.avi"),
        )
        failures: list[str] = []
        for codec, path in candidates:
            writer = self._writer_factory(path, codec, self.config.video_fps, self._frame_size)
            if writer is not None and bool(writer.isOpened()):
                self._writer = writer
                self._video_path = path
                self._video_codec = codec
                return
            if writer is not None:
                writer.release()
            failures.append(codec)
        raise RuntimeError(
            "Could not initialize diagnostic video writer; attempted codecs: "
            + ", ".join(failures)
        )

    def record_frame(
        self, frame: np.ndarray, *, capture_frame_index: int, timestamp: float
    ) -> bool:
        """Sample one original capture frame without resizing or overlay drawing."""
        if self._finalized:
            raise RuntimeError("Diagnostic evidence recorder is finalized")
        frame = validate_bgr_frame(frame)
        self._ensure_writer(frame)
        size = (frame.shape[1], frame.shape[0])
        if size != self._frame_size:
            raise RuntimeError(
                f"Diagnostic video frame size changed from {self._frame_size} to {size}"
            )
        self._last_capture_timestamp = float(timestamp)
        interval = 1.0 / self.config.video_fps
        if self._next_due is None:
            self._next_due = float(timestamp)
        if float(timestamp) + 1e-9 < self._next_due:
            return False
        skipped = max(0, int(math.floor((float(timestamp) - self._next_due) / interval + 1e-9)))
        if skipped:
            self._dropped_video_frames += skipped
            previous = self._video_samples[-1] if self._video_samples else None
            self._video_gaps.append({
                "after_video_frame_index": previous.video_frame_index if previous else None,
                "previous_timestamp": previous.timestamp if previous else None,
                "next_timestamp": float(timestamp),
                "missing_video_frames": skipped,
            })
        assert self._writer is not None
        self._writer.write(frame)
        sample = VideoSample(
            len(self._video_samples) + 1, int(capture_frame_index), float(timestamp)
        )
        self._video_samples.append(sample)
        with self.video_index_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=(
                "video_frame_index", "capture_frame_index", "timestamp",
            )).writerow({
                "video_frame_index": sample.video_frame_index,
                "capture_frame_index": sample.capture_frame_index,
                "timestamp": f"{sample.timestamp:.9f}",
            })
        self._next_due += (skipped + 1) * interval
        self._ring.append(sample)
        cutoff = sample.timestamp - self.config.event_pre_seconds
        while self._ring and self._ring[0].timestamp < cutoff:
            self._ring.popleft()
        for event in self._pending_events:
            if sample.timestamp <= event.event_timestamp + self.config.event_post_seconds + 1e-9:
                if not event.samples or event.samples[-1].video_frame_index != sample.video_frame_index:
                    event.samples.append(sample)
        self._finish_ready_events(sample.timestamp)
        return True

    def mark_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._finalized:
            return
        timestamp = float(payload.get("timestamp", self._last_capture_timestamp or 0.0))
        frame_index = int(payload.get("frame_index", 0) or 0)
        self._pending_events.append(PendingEventWindow(
            event_type=str(event_type),
            event_timestamp=timestamp,
            event_frame_index=frame_index,
            samples=list(self._ring),
        ))

    def _finish_ready_events(self, timestamp: float) -> None:
        keep: list[PendingEventWindow] = []
        for event in self._pending_events:
            if timestamp + 1e-9 >= event.event_timestamp + self.config.event_post_seconds:
                self._completed_events.append(self._event_row(event, complete=True))
            else:
                keep.append(event)
        self._pending_events = keep

    def _event_row(self, event: PendingEventWindow, *, complete: bool) -> dict[str, Any]:
        samples = sorted(
            {item.video_frame_index: item for item in event.samples}.values(),
            key=lambda item: item.video_frame_index,
        )
        first = samples[0] if samples else None
        last = samples[-1] if samples else None
        return {
            "event_type": event.event_type,
            "event_timestamp": event.event_timestamp,
            "event_capture_frame_index": event.event_frame_index,
            "requested_start_timestamp": max(0.0, event.event_timestamp - self.config.event_pre_seconds),
            "requested_end_timestamp": event.event_timestamp + self.config.event_post_seconds,
            "first_video_frame_index": first.video_frame_index if first else None,
            "last_video_frame_index": last.video_frame_index if last else None,
            "first_capture_frame_index": first.capture_frame_index if first else None,
            "last_capture_frame_index": last.capture_frame_index if last else None,
            "sample_count": len(samples),
            "post_window_complete": complete,
            "source": "full_session_video_ring_index",
        }

    def record_detector_evidence(
        self,
        frame: np.ndarray,
        *,
        capture_frame_index: int,
        timestamp: float,
        episode_id: int,
        runtime_state: str,
        roi_bounds: Mapping[str, tuple[int, int, int, int]],
        detector_metadata: Mapping[str, Any],
        executed: Mapping[str, bool],
    ) -> None:
        """Save four small ROIs for every frame on which any detector ran."""
        frame = validate_bgr_frame(frame)
        roi_paths: dict[str, str] = {}
        for name in ROI_NAMES:
            if name not in roi_bounds:
                raise KeyError(f"Missing diagnostic ROI bounds: {name}")
            x1, y1, x2, y2 = roi_bounds[name]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError(f"Diagnostic {name} ROI is empty: {roi_bounds[name]}")
            relative = Path("rois") / name / f"frame_{capture_frame_index:06d}.jpg"
            destination = self.root / relative
            if not self._image_writer(destination, crop, self.config.jpeg_quality):
                raise OSError(f"Could not write diagnostic ROI: {destination}")
            roi_paths[name] = relative.as_posix()
            self._roi_counts[int(episode_id)][name] += 1
            if bool(executed.get(name, False)):
                self._execution_counts[int(episode_id)][name] += 1
        row = {
            "capture_frame_index": int(capture_frame_index),
            "timestamp": float(timestamp),
            "episode_id": int(episode_id),
            "runtime_state": str(runtime_state),
            "executed": {name: bool(executed.get(name, False)) for name in ROI_NAMES},
            "roi_bounds": {name: list(roi_bounds[name]) for name in ROI_NAMES},
            "roi_paths": roi_paths,
            "detectors": _json_safe(detector_metadata),
        }
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._evidence_records += 1

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            return self.summary()
        for event in self._pending_events:
            self._completed_events.append(self._event_row(event, complete=False))
        self._pending_events.clear()
        with self.event_windows_path.open("w", encoding="utf-8") as handle:
            for row in self._completed_events:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._finalized = True
        return self.summary()

    def summary(self) -> dict[str, Any]:
        first = self._video_samples[0].timestamp if self._video_samples else None
        last = self._video_samples[-1].timestamp if self._video_samples else None
        gaps = list(self._video_gaps)
        if not self._video_samples:
            gaps.append({"reason": "no_video_frames"})
        if self._evidence_records == 0:
            gaps.append({"reason": "no_detector_roi_evidence"})
        return {
            "evidence_mode": "diagnostic",
            "video_path": str(self._video_path) if self._video_path else None,
            "video_codec": self._video_codec,
            "video_frame_count": len(self._video_samples),
            "video_fps": self.config.video_fps,
            "video_frame_size": list(self._frame_size) if self._frame_size else None,
            "first_timestamp": first,
            "last_timestamp": last,
            "dropped_video_frames": self._dropped_video_frames,
            "video_index_path": str(self.video_index_path),
            "detector_evidence_path": str(self.metadata_path),
            "event_windows_path": str(self.event_windows_path),
            "event_window_count": len(self._completed_events),
            "roi_evidence_counts_by_episode": {
                str(episode): dict(counts) for episode, counts in sorted(self._roi_counts.items())
            },
            "detector_execution_counts_by_episode": {
                str(episode): dict(counts)
                for episode, counts in sorted(self._execution_counts.items())
            },
            "evidence_record_count": self._evidence_records,
            "has_evidence_gaps": bool(gaps),
            "evidence_gap_intervals": gaps,
        }

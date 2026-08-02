"""Bounded, opt-in PRESS anomaly evidence without continuous recording."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.fishing_v2.domain.observations import PressObservation


@dataclass(frozen=True)
class PressAnomalyEvidenceConfig:
    enabled: bool = False
    buffer_frames: int = 12
    max_episodes: int = 20
    frames_per_anomaly: int = 6

    def __post_init__(self) -> None:
        if self.buffer_frames < 1 or self.max_episodes < 1:
            raise ValueError("PRESS anomaly evidence bounds must be positive")
        if not 1 <= self.frames_per_anomaly <= self.buffer_frames:
            raise ValueError("frames_per_anomaly must fit the PRESS anomaly buffer")


@dataclass(frozen=True)
class _Sample:
    episode_index: int
    frame_index: int
    timestamp: float
    pixels: np.ndarray
    diagnostics: Mapping[str, Any]


class PressAnomalyEvidenceRecorder:
    """Keep ROI samples in RAM and write only explicitly triggered anomalies."""

    def __init__(
        self,
        session_path: str | Path,
        config: PressAnomalyEvidenceConfig | None = None,
    ) -> None:
        self.config = config or PressAnomalyEvidenceConfig()
        self.root = Path(session_path) / "press_anomalies"
        self._samples: deque[_Sample] = deque(maxlen=self.config.buffer_frames)
        self._triggered: set[int] = set()
        self._futures: list[Future[Any]] = []
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="press-anomaly-writer")
            if self.config.enabled else None
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def record(
        self,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        roi_pixels: np.ndarray | None,
        observation: PressObservation | None,
        certificate: Mapping[str, Any] | None,
    ) -> None:
        if not self.enabled or roi_pixels is None or observation is None:
            return
        self._samples.append(_Sample(
            int(episode_index),
            int(frame_index),
            float(timestamp),
            roi_pixels.copy(),
            {
                "episode_index": int(episode_index),
                "capture_frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "panel_present": bool(observation.panel_present),
                "panel_candidate": bool(observation.panel_candidate),
                "sequence_candidate": list(observation.sequence_candidate),
                "sequence_confidence": float(observation.sequence_confidence),
                "panel_phase": observation.evidence.get("panel_phase"),
                "slots": list(observation.evidence.get("slots", ())),
                "press_completeness_certificate": dict(certificate or {}),
            },
        ))

    def trigger(self, *, episode_index: int, reason: str) -> bool:
        if (
            not self.enabled
            or self._executor is None
            or episode_index in self._triggered
            or len(self._triggered) >= self.config.max_episodes
        ):
            return False
        selected = [
            sample for sample in self._samples
            if sample.episode_index == int(episode_index)
        ][-self.config.frames_per_anomaly:]
        if not selected:
            return False
        self._triggered.add(int(episode_index))
        # Copy the bounded snapshot before returning to the capture loop.
        snapshot = tuple(selected)
        self._futures.append(self._executor.submit(
            self._write_episode,
            int(episode_index),
            str(reason),
            snapshot,
        ))
        return True

    @staticmethod
    def _draw_occupied(sample: _Sample) -> np.ndarray:
        image = sample.pixels.copy()
        for slot in sample.diagnostics.get("slots", ()):
            if not isinstance(slot, Mapping):
                continue
            bbox = slot.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            occupancy = str(slot.get("occupancy", "UNCERTAIN"))
            colour = {
                "OCCUPIED": (0, 220, 0),
                "EMPTY": (120, 120, 120),
                "UNCERTAIN": (0, 170, 255),
            }.get(occupancy, (0, 0, 255))
            x1, y1, x2, y2 = (int(value) for value in bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                image, occupancy, (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
            )
        return image

    @staticmethod
    def _draw_arrows(sample: _Sample) -> np.ndarray:
        image = sample.pixels.copy()
        for slot in sample.diagnostics.get("slots", ()):
            if not isinstance(slot, Mapping):
                continue
            bbox = slot.get("arrow_bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(value) for value in bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (230, 80, 230), 2)
            key = str(slot.get("mapped_key") or "?")
            cv2.putText(
                image, key, (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 80, 230), 1,
            )
        return image

    def _write_episode(
        self,
        episode_index: int,
        reason: str,
        samples: tuple[_Sample, ...],
    ) -> None:
        destination = self.root / f"episode_{episode_index}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "anomaly.json").write_text(
            json.dumps({
                "episode_index": episode_index,
                "reason": reason,
                "frame_count": len(samples),
                "frames": [sample.frame_index for sample in samples],
            }, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for sample in samples:
            stem = f"frame_{sample.frame_index:06d}"
            cv2.imwrite(str(destination / f"{stem}_raw.png"), sample.pixels)
            cv2.imwrite(
                str(destination / f"{stem}_occupied.png"),
                self._draw_occupied(sample),
            )
            cv2.imwrite(
                str(destination / f"{stem}_arrows.png"),
                self._draw_arrows(sample),
            )
            (destination / f"{stem}.json").write_text(
                json.dumps(sample.diagnostics, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def close(self) -> dict[str, Any]:
        failures: list[str] = []
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            for future in self._futures:
                try:
                    future.result()
                except Exception as exc:  # evidence must never fail Runtime
                    failures.append(f"{type(exc).__name__}: {exc}")
        return {
            "press_anomaly_evidence_enabled": self.enabled,
            "press_anomaly_evidence_path": (
                str(self.root) if self._triggered else None
            ),
            "press_anomaly_episode_count": len(self._triggered),
            "press_anomaly_evidence_failures": failures,
            "press_anomaly_video_writer_initialized": False,
            "press_anomaly_mp4_created": False,
        }

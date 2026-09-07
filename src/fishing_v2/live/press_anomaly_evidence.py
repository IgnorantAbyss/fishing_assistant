"""Bounded, opt-in PRESS anomaly evidence without continuous recording."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.fishing_v2.domain.observations import PressObservation


def _evidence_only(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            self._disabled = True
            return False
    return guarded


@dataclass(frozen=True)
class PressAnomalyEvidenceConfig:
    enabled: bool = False
    buffer_frames: int = 12
    max_episodes: int = 20
    frames_per_anomaly: int = 6
    pending_seconds: float = 3.0  # Diagnostic only, never an action timeout.
    sample_interval_seconds: float = 0.2
    max_roi_bytes: int = 1024 * 1024
    max_metadata_bytes: int = 65536

    def __post_init__(self) -> None:
        if self.buffer_frames < 1 or self.max_episodes < 1:
            raise ValueError("PRESS anomaly evidence bounds must be positive")
        if not 1 <= self.frames_per_anomaly <= self.buffer_frames:
            raise ValueError("frames_per_anomaly must fit the PRESS anomaly buffer")
        if self.pending_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise ValueError("diagnostic durations must be positive")


@dataclass(frozen=True)
class _Sample:
    episode_index: int
    frame_index: int
    timestamp: float
    pixels: np.ndarray
    metadata: bytes

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return json.loads(self.metadata)


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
        self._pinned: dict[int, dict[str, _Sample]] = {}
        self._triggered: set[int] = set()
        self._futures: list[Future[Any]] = []
        self._failure = None
        self._disabled = False
        self._active_episode = None
        self._pending_since = None
        self._pending_episode = None
        self._bucket = None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="press-anomaly-writer")
            if self.config.enabled else None
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._disabled

    @staticmethod
    def incomplete_episode_eligible(
        authoritative_progress: Mapping[str, Any] | None,
    ) -> bool:
        progress = dict(authoritative_progress or {})
        return not any(
            bool(progress.get(boundary))
            for boundary in (
                "sequence_frozen",
                "opportunity_created",
                "opportunity_scheduled",
                "emission_started",
                "action_applied",
            )
        )

    @_evidence_only
    def record(
        self,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        roi_pixels: np.ndarray | None,
        observation: PressObservation | None,
        certificate: Mapping[str, Any] | None,
        runtime_state: str | None = None,
        authoritative_progress: Mapping[str, Any] | None = None,
        frozen_sequence: tuple[str, ...] = (),
        qualified_detected: bool | None = None,
    ) -> None:
        if not self.enabled or roi_pixels is None or observation is None:
            return
        if episode_index in self._triggered or len(self._triggered) >= self.config.max_episodes:
            return
        if roi_pixels.nbytes > self.config.max_roi_bytes:
            raise ValueError("PRESS evidence ROI exceeds byte cap")
        if self._active_episode != episode_index:
            self._samples.clear()
            self._pinned.clear()
            self._bucket = None
            self._active_episode = episode_index
        sample = _Sample(
            int(episode_index),
            int(frame_index),
            float(timestamp),
            roi_pixels.copy(),
            json.dumps({
                "episode_index": int(episode_index),
                "capture_frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "raw_roi_shape": list(roi_pixels.shape),
                "runtime_state": runtime_state,
                "panel_bbox": observation.evidence.get("panel_bbox"),
                "panel_identity": observation.evidence.get("press_episode_id"),
                "press_evidence_version": observation.evidence.get("press_evidence_version"),
                "authoritative_progress": dict(authoritative_progress or {}),
                "frozen_sequence": list(frozen_sequence),
                "qualified_detected": qualified_detected,
                "input_effect": {key: observation.evidence.get(key) for key in (
                    "input_effect_detected", "input_started", "post_input_frame",
                    "frame_structurally_complete", "clean_frame_eligible")},
                "panel_present": bool(observation.panel_present),
                "panel_candidate": bool(observation.panel_candidate),
                "sequence_candidate": list(observation.sequence_candidate),
                "sequence_confidence": float(observation.sequence_confidence),
                "panel_phase": observation.evidence.get("panel_phase"),
                "slots": list(observation.evidence.get("slots", ())),
                "geometry_continuity": dict(
                    observation.evidence.get("v3", {}).get(
                        "geometry_continuity", {}
                    )
                ),
                "press_completeness_certificate": dict(certificate or {}),
            }, ensure_ascii=False).encode("utf-8"),
        )
        if len(sample.metadata) > self.config.max_metadata_bytes:
            raise ValueError("PRESS evidence metadata exceeds byte cap")
        # Latest frame in each time bucket, never a duplicate synthetic frame.
        bucket = int(timestamp / self.config.sample_interval_seconds)
        if self._samples and bucket == self._bucket:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        self._bucket = bucket
        pins = self._pinned.setdefault(int(episode_index), {})
        completeness = dict(certificate or {})
        if observation.panel_present:
            pins.setdefault("first_panel_present", sample)
        if qualified_detected is True:
            pins.setdefault("first_strong_evidence", sample)
        if completeness.get("complete") is False:
            pins.setdefault("first_completeness_pending", sample)
        if (completeness.get("decoded_count", 0) > 0
                and completeness.get("decoded_count") == completeness.get("occupied_count")):
            pins.setdefault("first_fully_decoded", sample)
        if (completeness.get("panel_bbox_stable") is False
                or sample.diagnostics["geometry_continuity"].get("rejection_reason")):
            pins.setdefault("first_geometry_rejection", sample)
        sequence_stability = int(
            completeness.get("sequence_stability_count", 0) or 0
        )
        if (
            completeness.get("complete") is False
            and sequence_stability >= 2
            and observation.sequence_candidate
        ):
            pins.setdefault("first_incomplete_stable_candidate", sample)
        decoded_count = int(completeness.get("decoded_count", 0) or 0)
        best = pins.get("best_decoded_candidate")
        best_decoded = (
            int(best.diagnostics["press_completeness_certificate"].get(
                "decoded_count", 0
            ) or 0)
            if best is not None else -1
        )
        if decoded_count > best_decoded:
            pins["best_decoded_candidate"] = sample
        occupied_count = int(completeness.get("occupied_count", 0) or 0)
        if occupied_count != decoded_count:
            pins.setdefault("first_occupancy_decoded_mismatch", sample)

    @_evidence_only
    def trigger(self, *, episode_index: int, reason: str) -> bool:
        if (
            not self.enabled
            or self._executor is None
            or episode_index in self._triggered
            or len(self._triggered) >= self.config.max_episodes
        ):
            return False
        recent = [
            sample for sample in self._samples
            if sample.episode_index == int(episode_index)
        ]
        if not recent:
            return False
        pins = dict(self._pinned.get(int(episode_index), {}))
        pins["last_before_panel_disappears"] = recent[-1]
        selected_by_frame: dict[int, _Sample] = {}
        for category in (
            "first_incomplete_stable_candidate",
            "best_decoded_candidate",
            "first_occupancy_decoded_mismatch",
            "last_before_panel_disappears",
            "first_panel_present", "first_strong_evidence",
            "first_fully_decoded", "first_completeness_pending",
            "first_geometry_rejection",
        ):
            if len(selected_by_frame) >= self.config.frames_per_anomaly:
                break
            sample = pins.get(category)
            if sample is not None:
                selected_by_frame.setdefault(sample.frame_index, sample)
        for sample in reversed(recent):
            if len(selected_by_frame) >= self.config.frames_per_anomaly:
                break
            selected_by_frame.setdefault(sample.frame_index, sample)
        selected = sorted(selected_by_frame.values(), key=lambda item: item.frame_index)
        selected = selected[:self.config.frames_per_anomaly]
        self._triggered.add(int(episode_index))
        # Copy the bounded snapshot before returning to the capture loop.
        snapshot = tuple(selected)
        self._futures.append(self._executor.submit(
            self._write_guarded,
            int(episode_index),
            str(reason),
            snapshot,
            {
                category: sample.frame_index
                for category, sample in pins.items()
            },
        ))
        return True

    def _write_guarded(self, *args):
        if self._disabled:
            return
        try:
            self._write_episode(*args)
        except Exception:
            self._disabled = True
            raise

    @_evidence_only
    def service_pending(self, *, episode_index: int, timestamp: float,
                        runtime_state: str, certificate: Mapping[str, Any] | None,
                        authoritative_progress: Mapping[str, Any] | None,
                        shutdown: bool = False) -> bool:
        """Observe only. No lifecycle mutation, guessed sequence or action callback."""
        if not self.enabled:
            return False
        if (runtime_state != "PRESS"
                or not self.incomplete_episode_eligible(authoritative_progress)
                or (certificate is not None and certificate.get("complete") is True)):
            self._pending_since = self._pending_episode = None
            return False
        if self._pending_episode != episode_index:
            self._pending_since = self._pending_episode = None
        if self._pending_since is None:
            if not certificate or certificate.get("complete") is not False:
                return False
            self._pending_episode, self._pending_since = episode_index, timestamp
        if timestamp - self._pending_since < self.config.pending_seconds:
            return False
        return self.trigger_incomplete(
            episode_index=episode_index,
            reason=("press_pending_at_shutdown" if shutdown else "press_completeness_prolonged_pending"),
            authoritative_progress=authoritative_progress)

    def trigger_incomplete(
        self,
        *,
        episode_index: int,
        reason: str,
        authoritative_progress: Mapping[str, Any] | None,
    ) -> bool:
        """Write only genuine pre-freeze incomplete PRESS episodes."""
        if not self.incomplete_episode_eligible(authoritative_progress):
            return False
        return self.trigger(episode_index=episode_index, reason=reason)

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
        categories: Mapping[str, int],
    ) -> None:
        if self._disabled:
            return
        destination = self.root / f"episode_{episode_index}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "anomaly.json").write_text(
            json.dumps({
                "episode_index": episode_index,
                "reason": reason,
                "frame_count": len(samples),
                "frames": [sample.frame_index for sample in samples],
                "evidence_categories": dict(categories),
            }, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for sample in samples:
            stem = f"frame_{sample.frame_index:06d}"
            if not cv2.imwrite(str(destination / f"{stem}_raw.png"), sample.pixels):
                raise OSError("PRESS raw evidence image write failed")
            cv2.imwrite(
                str(destination / f"{stem}_occupied.png"),
                self._draw_occupied(sample),
            )
            cv2.imwrite(
                str(destination / f"{stem}_arrows.png"),
                self._draw_arrows(sample),
            )
            (destination / f"{stem}.json").write_bytes(sample.metadata)

    def close(self) -> dict[str, Any]:
        failures: list[str] = [self._failure] if self._failure else []
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
            "press_anomaly_pending_diagnostic_seconds": self.config.pending_seconds,
            "press_anomaly_sample_interval_seconds": self.config.sample_interval_seconds,
            "press_anomaly_buffer_frames": self.config.buffer_frames,
            "press_anomaly_pin_cap": 8,
            "press_anomaly_frames_per_episode": self.config.frames_per_anomaly,
            "press_anomaly_max_episodes": self.config.max_episodes,
            "press_anomaly_max_roi_bytes": self.config.max_roi_bytes,
            "press_anomaly_max_metadata_bytes": self.config.max_metadata_bytes,
            "press_anomaly_evidence_failures": failures,
            "press_anomaly_video_writer_initialized": False,
            "press_anomaly_mp4_created": False,
        }

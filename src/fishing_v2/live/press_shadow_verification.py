"""Non-blocking PRESS sequence shadow verification and deferred diagnostics."""

from __future__ import annotations

from collections import Counter, deque
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import cv2
import numpy as np

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode


TRACE_NAME = "press_decision_trace.jsonl"
CLIP_NAME = "press_roi_clip.mp4"
CLIP_INDEX_NAME = "press_roi_frames.csv"
REVIEW_NAME = "press_review_items.csv"
REVIEW_FIELDS = (
    "episode_index",
    "predicted_sequence",
    "sequence_length",
    "first_detected_timestamp",
    "frozen_timestamp",
    "proposal_timestamp",
    "detector_frame_count",
    "actual_detector_fps",
    "stability_frame_count",
    "rejection_reasons",
    "exactly_once_proposal_count",
)


@dataclass(frozen=True)
class PressShadowProposal:
    episode_index: int
    sequence: tuple[str, ...]
    timestamp: float
    frame_index: int
    stability_frame_count: int


@dataclass(frozen=True)
class _PressROISample:
    episode_index: int
    frame_index: int
    timestamp: float
    pixels: np.ndarray


class PressShadowVerifier:
    """Keep PRESS decisions in memory and emit no operating-system input."""

    def __init__(
        self,
        *,
        max_trace_entries_per_episode: int = 512,
        max_session_trace_entries: int = 4096,
        max_roi_frames: int = 4096,
        stability_history_frames: int = 5,
    ) -> None:
        if max_trace_entries_per_episode < 1:
            raise ValueError("max_trace_entries_per_episode must be positive")
        if max_session_trace_entries < max_trace_entries_per_episode:
            raise ValueError("max_session_trace_entries must fit one episode")
        if max_roi_frames < 1 or stability_history_frames < 1:
            raise ValueError("PRESS shadow buffer sizes must be positive")
        self._max_trace_entries = max_trace_entries_per_episode
        self._completed_trace: deque[dict[str, Any]] = deque(
            maxlen=max_session_trace_entries
        )
        self._roi_samples: deque[_PressROISample] = deque(maxlen=max_roi_frames)
        self._history_size = stability_history_frames
        self._episode_index = 0
        self._active = False
        self._trace: deque[dict[str, Any]] = deque(
            maxlen=max_trace_entries_per_episode
        )
        self._stability_history: deque[str] = deque(
            maxlen=stability_history_frames
        )
        self._detector_timestamps: list[float] = []
        self._rejections: Counter[str] = Counter()
        self._first_detected_timestamp: float | None = None
        self._frozen_timestamp: float | None = None
        self._proposal_timestamp: float | None = None
        self._frozen_sequence: tuple[str, ...] = ()
        self._proposal_count = 0
        self._proposal_created = False
        self._stability_frame_count = 0
        self._review_items: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return self._active

    @property
    def proposal_count(self) -> int:
        return sum(
            int(row["exactly_once_proposal_count"])
            for row in self._review_items
        ) + self._proposal_count

    @staticmethod
    def _normalize(sequence: Any) -> tuple[str, ...]:
        if not isinstance(sequence, (list, tuple)):
            return ()
        return tuple(
            value
            for value in (str(item).strip().upper() for item in sequence)
            if value in {"W", "A", "S", "D"}
        )

    @staticmethod
    def _bbox(item: Mapping[str, Any]) -> list[float] | None:
        value = item.get("bbox")
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            return [float(component) for component in value]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _raw_symbols(
        cls,
        observation: PressObservation | None,
    ) -> list[dict[str, Any]]:
        if observation is None:
            return []
        evidence = observation.evidence
        entries: list[dict[str, Any]] = []
        slots = evidence.get("slots", ())
        if isinstance(slots, (list, tuple)):
            for item in slots:
                if not isinstance(item, Mapping):
                    continue
                symbol = str(
                    item.get("arrow_direction", item.get("key", ""))
                ).strip().upper()
                bbox = cls._bbox(item)
                if (
                    symbol in {"W", "A", "S", "D"}
                    and bbox is not None
                    and item.get("occupancy", "OCCUPIED") == "OCCUPIED"
                ):
                    entries.append({
                        "symbol": symbol,
                        "bbox": bbox,
                        "confidence": float(
                            item.get(
                                "arrow_confidence",
                                item.get("confidence", 0.0),
                            )
                        ),
                    })
        if not entries:
            boxes = evidence.get("key_boxes", ())
            if isinstance(boxes, (list, tuple)):
                for item in boxes:
                    if not isinstance(item, Mapping):
                        continue
                    symbol = str(item.get("key", "")).strip().upper()
                    bbox = cls._bbox(item)
                    if symbol in {"W", "A", "S", "D"} and bbox is not None:
                        entries.append({
                            "symbol": symbol,
                            "bbox": bbox,
                            "confidence": float(item.get("confidence", 0.0)),
                        })
        entries.sort(
            key=lambda item: (
                (item["bbox"][0] + item["bbox"][2]) * 0.5,
                item["bbox"][1],
            )
        )
        return entries

    def _start_episode(self) -> None:
        self._episode_index += 1
        self._active = True
        self._trace.clear()
        self._stability_history.clear()
        self._detector_timestamps = []
        self._rejections.clear()
        self._first_detected_timestamp = None
        self._frozen_timestamp = None
        self._proposal_timestamp = None
        self._frozen_sequence = ()
        self._proposal_count = 0
        self._proposal_created = False
        self._stability_frame_count = 0

    def _finish_episode(self, reason: str) -> None:
        if not self._active:
            return
        for row in self._trace:
            self._completed_trace.append({
                **row,
                "episode_end_reason": reason,
            })
        duration = (
            self._detector_timestamps[-1] - self._detector_timestamps[0]
            if len(self._detector_timestamps) > 1 else 0.0
        )
        actual_fps = (
            (len(self._detector_timestamps) - 1) / duration
            if duration > 0.0 else 0.0
        )
        self._review_items.append({
            "episode_index": self._episode_index,
            "predicted_sequence": "".join(self._frozen_sequence),
            "sequence_length": len(self._frozen_sequence),
            "first_detected_timestamp": self._first_detected_timestamp,
            "frozen_timestamp": self._frozen_timestamp,
            "proposal_timestamp": self._proposal_timestamp,
            "detector_frame_count": len(self._detector_timestamps),
            "actual_detector_fps": round(actual_fps, 4),
            "stability_frame_count": self._stability_frame_count,
            "rejection_reasons": json.dumps(
                dict(self._rejections),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "exactly_once_proposal_count": self._proposal_count,
        })
        self._active = False
        self._trace.clear()
        self._stability_history.clear()
        self._detector_timestamps = []
        self._rejections.clear()
        self._frozen_sequence = ()
        self._proposal_count = 0
        self._proposal_created = False

    def observe(
        self,
        *,
        timestamp: float,
        frame_index: int,
        fsm_state: RuntimeState,
        activation_mode: DetectorActivationMode,
        raw: PressObservation | None,
        qualified: PressObservation | None,
        qualification_reason: str,
        safety_reason: str,
        roi_pixels: np.ndarray | None = None,
    ) -> PressShadowProposal | None:
        if fsm_state != RuntimeState.PRESS:
            self._finish_episode("runtime_left_press")
            return None
        if not self._active:
            self._start_episode()
        if raw is None:
            return None

        self._detector_timestamps.append(float(timestamp))
        raw_symbols = self._raw_symbols(raw)
        sorted_sequence = tuple(item["symbol"] for item in raw_symbols)
        if not sorted_sequence:
            sorted_sequence = self._normalize(raw.sequence_candidate)
        normalized = self._normalize(sorted_sequence)
        self._stability_history.append("".join(normalized))
        if raw.panel_present and self._first_detected_timestamp is None:
            self._first_detected_timestamp = float(timestamp)
        if roi_pixels is not None:
            self._roi_samples.append(_PressROISample(
                self._episode_index,
                int(frame_index),
                float(timestamp),
                roi_pixels.copy(),
            ))

        frozen = self._normalize(
            qualified.sequence if qualified is not None else ()
        )
        if not frozen and qualified is not None and qualified.sequence_ready:
            frozen = self._normalize(qualified.sequence_candidate)
        if frozen and not self._frozen_sequence:
            self._frozen_sequence = frozen
            self._frozen_timestamp = float(timestamp)
            frozen_text = "".join(frozen)
            self._stability_frame_count = sum(
                value == frozen_text for value in self._stability_history
            )

        rejection: str | None = None
        if activation_mode != DetectorActivationMode.ACTIVE:
            rejection = "press_detector_not_active"
        elif qualified is None or not qualified.detected:
            rejection = "qualified_press_panel_absent"
        elif not qualified.sequence_ready or not frozen:
            rejection = qualification_reason or "stable_sequence_not_ready"
        elif self._proposal_created:
            rejection = "press_sequence_opportunity_already_consumed"

        proposal: PressShadowProposal | None = None
        one_shot_blocked = rejection == "press_sequence_opportunity_already_consumed"
        if rejection is None:
            self._proposal_created = True
            self._proposal_count += 1
            self._proposal_timestamp = float(timestamp)
            self._frozen_sequence = frozen
            frozen_text = "".join(frozen)
            self._stability_frame_count = max(
                self._stability_frame_count,
                sum(value == frozen_text for value in self._stability_history),
            )
            proposal = PressShadowProposal(
                self._episode_index,
                frozen,
                float(timestamp),
                int(frame_index),
                self._stability_frame_count,
            )
        else:
            self._rejections[rejection] += 1

        self._trace.append({
            "timestamp": float(timestamp),
            "capture_frame_index": int(frame_index),
            "fsm_state": fsm_state.value,
            "press_episode_active": self._active,
            "activation_mode": activation_mode.value,
            "raw_detected": bool(raw.detected),
            "raw_detected_symbols": raw_symbols,
            "left_to_right_sequence": list(sorted_sequence),
            "normalized_sequence": list(normalized),
            "sequence_length": len(normalized),
            "stability_history": list(self._stability_history),
            "candidate_accepted": proposal is not None,
            "candidate_rejected_reason": rejection,
            "qualification_reason": qualification_reason,
            "frozen_sequence": list(self._frozen_sequence),
            "proposal_created": proposal is not None,
            "one_shot_blocked": one_shot_blocked,
            "safety_reason": safety_reason,
            "allowlist_rejection_reason": (
                "press_sequence_shadow_only_not_live_allowlisted"
            ),
            "action_sink_called": False,
        })

        if bool(
            qualified
            and qualified.evidence.get("panel_disappeared")
        ):
            self._finish_episode("press_panel_disappeared")
        return proposal

    def finish_session(self) -> None:
        self._finish_episode("session_ended")

    @staticmethod
    def _clip_fps(samples: list[_PressROISample]) -> float:
        intervals = [
            current.timestamp - previous.timestamp
            for previous, current in zip(samples, samples[1:])
            if current.timestamp > previous.timestamp
        ]
        if not intervals:
            return 0.0
        return round(min(30.0, 1.0 / median(intervals)), 4)

    def write_artifacts(self, output_dir: str | Path) -> dict[str, Any]:
        """Write deferred evidence after the capture loop has stopped."""
        self.finish_session()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        trace_path = destination / TRACE_NAME
        with trace_path.open("w", encoding="utf-8") as handle:
            for row in self._completed_trace:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

        review_path = destination / REVIEW_NAME
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(self._review_items)

        samples = list(self._roi_samples)
        index_path = destination / CLIP_INDEX_NAME
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "clip_frame_index",
                "episode_index",
                "capture_frame_index",
                "timestamp",
            ))
            writer.writeheader()
            for index, sample in enumerate(samples, start=1):
                writer.writerow({
                    "clip_frame_index": index,
                    "episode_index": sample.episode_index,
                    "capture_frame_index": sample.frame_index,
                    "timestamp": sample.timestamp,
                })

        video_path = destination / CLIP_NAME
        output_fps = self._clip_fps(samples)
        video_written = False
        if samples and output_fps > 0.0:
            height, width = samples[0].pixels.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )
            if writer.isOpened():
                try:
                    for sample in samples:
                        if sample.pixels.shape[:2] == (height, width):
                            writer.write(sample.pixels)
                    video_written = True
                finally:
                    writer.release()
            else:
                writer.release()

        return {
            "press_decision_trace_path": str(trace_path),
            "press_decision_trace_rows": len(self._completed_trace),
            "press_roi_clip_path": str(video_path) if video_written else None,
            "press_roi_frames_path": str(index_path),
            "press_roi_frame_count": len(samples),
            "press_roi_clip_fps": output_fps if video_written else 0.0,
            "press_review_items_path": str(review_path),
            "press_episode_review_items": list(self._review_items),
            "press_shadow_proposal_count": sum(
                int(row["exactly_once_proposal_count"])
                for row in self._review_items
            ),
        }

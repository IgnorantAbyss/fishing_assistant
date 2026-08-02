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
from src.fishing_v2.runtime.press_action_contract import (
    validate_frozen_press_payload,
)


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
    slot_capacity: int


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
        roi_pre_roll_seconds: float = 1.0,
        roi_post_roll_seconds: float = 0.5,
        diagnostics_enabled: bool = True,
    ) -> None:
        if max_trace_entries_per_episode < 1:
            raise ValueError("max_trace_entries_per_episode must be positive")
        if max_session_trace_entries < max_trace_entries_per_episode:
            raise ValueError("max_session_trace_entries must fit one episode")
        if max_roi_frames < 1 or stability_history_frames < 1:
            raise ValueError("PRESS shadow buffer sizes must be positive")
        if roi_pre_roll_seconds < 0 or roi_post_roll_seconds < 0:
            raise ValueError("PRESS ROI roll durations must be non-negative")
        self._max_trace_entries = max_trace_entries_per_episode
        self._completed_trace: deque[dict[str, Any]] = deque(
            maxlen=max_session_trace_entries if diagnostics_enabled else 1
        )
        self._roi_samples: deque[_PressROISample] | None = (
            deque(maxlen=max_roi_frames) if diagnostics_enabled else None
        )
        self._roi_pre_roll: deque[_PressROISample] | None = (
            deque(maxlen=256) if diagnostics_enabled else None
        )
        self._roi_pre_roll_seconds = float(roi_pre_roll_seconds)
        self._roi_post_roll_seconds = float(roi_post_roll_seconds)
        self._diagnostics_enabled = bool(diagnostics_enabled)
        self._roi_capture_episode: int | None = None
        self._roi_waiting_for_disappearance = False
        self._roi_post_roll_until: float | None = None
        self._dropped_roi_frames = 0
        self._roi_markers: dict[tuple[int, str], tuple[int, float]] = {}
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

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def frozen_sequence(self) -> tuple[str, ...]:
        return self._frozen_sequence

    def cancel_for_authoritative_idle_recovery(self) -> None:
        if self._active:
            self._finish_episode("authoritative_idle_recovery")

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

    @staticmethod
    def _slot_capacity(
        observation: PressObservation | None,
    ) -> int:
        if observation is None:
            return 0
        value = observation.evidence.get("total_slot_count")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        slots = observation.evidence.get("slots")
        return len(slots) if isinstance(slots, (list, tuple)) else 0

    def _append_roi_sample(self, sample: _PressROISample) -> None:
        if self._roi_samples is None:
            return
        if len(self._roi_samples) == self._roi_samples.maxlen:
            self._dropped_roi_frames += 1
        self._roi_samples.append(sample)

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
        self._roi_capture_episode = self._episode_index
        self._roi_waiting_for_disappearance = True
        self._roi_post_roll_until = None
        if self._roi_pre_roll is not None:
            for sample in self._roi_pre_roll:
                self._append_roi_sample(_PressROISample(
                    self._episode_index,
                    sample.frame_index,
                    sample.timestamp,
                    sample.pixels,
                ))
            self._roi_pre_roll.clear()

    def capture_roi_frame(
        self,
        *,
        timestamp: float,
        frame_index: int,
        fsm_state: RuntimeState,
        activation_mode: DetectorActivationMode,
        roi_pixels: np.ndarray | None,
        panel_disappeared: bool = False,
    ) -> None:
        """Retain real-timestamp pre/post-roll in memory without disk I/O."""
        if not self._diagnostics_enabled or roi_pixels is None:
            return
        pixels = roi_pixels.copy()
        if self._roi_capture_episode is None:
            if (
                fsm_state == RuntimeState.RESULT_PENDING
                and activation_mode == DetectorActivationMode.ARMED
            ):
                assert self._roi_pre_roll is not None
                self._roi_pre_roll.append(_PressROISample(
                    0,
                    int(frame_index),
                    float(timestamp),
                    pixels,
                ))
                cutoff = float(timestamp) - self._roi_pre_roll_seconds
                while (
                    self._roi_pre_roll
                    and self._roi_pre_roll[0].timestamp < cutoff
                ):
                    self._roi_pre_roll.popleft()
            return
        if panel_disappeared and self._roi_waiting_for_disappearance:
            self._roi_waiting_for_disappearance = False
            self._roi_post_roll_until = (
                float(timestamp) + self._roi_post_roll_seconds
            )
        if (
            self._roi_waiting_for_disappearance
            or self._roi_post_roll_until is None
            or float(timestamp) <= self._roi_post_roll_until + 1e-9
        ):
            self._append_roi_sample(_PressROISample(
                self._roi_capture_episode,
                int(frame_index),
                float(timestamp),
                pixels,
            ))
        elif float(timestamp) > self._roi_post_roll_until:
            self._roi_capture_episode = None
            self._roi_post_roll_until = None

    def _finish_episode(self, reason: str) -> None:
        if not self._active:
            return
        if self._diagnostics_enabled:
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
        live_emission_enabled: bool = False,
    ) -> PressShadowProposal | None:
        if fsm_state != RuntimeState.PRESS:
            self._finish_episode("runtime_left_press")
            return None
        if not self._active:
            self._start_episode()
        if roi_pixels is not None:
            self.capture_roi_frame(
                timestamp=timestamp,
                frame_index=frame_index,
                fsm_state=fsm_state,
                activation_mode=activation_mode,
                roi_pixels=roi_pixels,
                panel_disappeared=bool(
                    qualified
                    and qualified.evidence.get("panel_disappeared")
                ),
            )
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
        frozen = tuple(
            qualified.sequence if qualified is not None else ()
        )
        if not frozen and qualified is not None and qualified.sequence_ready:
            frozen = tuple(qualified.sequence_candidate)
        slot_capacity = self._slot_capacity(raw)
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
        else:
            try:
                validate_frozen_press_payload({
                    "sequence": frozen,
                    "slot_capacity": slot_capacity,
                    "active_press_episode": True,
                    "panel_confirmed": True,
                    "frozen_by_consensus": True,
                })
            except ValueError as exc:
                rejection = str(exc)
        if rejection is None and self._proposal_created:
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
                slot_capacity,
            )
            self._roi_markers[(self._episode_index, "frozen")] = (
                int(frame_index),
                float(timestamp),
            )
        else:
            self._rejections[rejection] += 1

        if self._diagnostics_enabled:
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
                    None
                    if live_emission_enabled
                    else "press_sequence_shadow_only_not_live_allowlisted"
                ),
                "action_sink_called": False,
            })

        if bool(
            qualified
            and qualified.evidence.get("panel_disappeared")
        ):
            self._finish_episode("press_panel_disappeared")
        return proposal

    def record_emission_result(
        self,
        *,
        safety_reason: str,
        action_sink_called: bool,
        terminal_outcome: str,
        attempted_count: int = 0,
        completed_key_count: int = 0,
        total_key_count: int = 0,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> None:
        """Attach Live outcome in memory without writing diagnostic files."""
        if not self._trace:
            return
        self._trace[-1].update({
            "safety_reason": safety_reason,
            "allowlist_rejection_reason": None,
            "action_sink_called": action_sink_called,
            "emission_terminal_outcome": terminal_outcome,
            "attempted_count": attempted_count,
            "completed_key_count": completed_key_count,
            "total_key_count": total_key_count,
        })
        if action_sink_called and timestamp is not None and frame_index is not None:
            self._roi_markers[(self._episode_index, "emission")] = (
                int(frame_index),
                float(timestamp),
            )

    def record_schedule(
        self,
        *,
        deadline: float,
        timing: Mapping[str, Any],
    ) -> None:
        if self._trace:
            self._trace[-1].update({
                "scheduled_emission_deadline": float(deadline),
                "sampled_press_timing": dict(timing),
            })

    def record_visual_ack(
        self,
        *,
        timestamp: float,
        frame_index: int,
    ) -> None:
        self._roi_markers[(self._episode_index, "visual_ack")] = (
            int(frame_index),
            float(timestamp),
        )

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

        samples = list(self._roi_samples or ())
        index_path = destination / CLIP_INDEX_NAME
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "clip_frame_index",
                "episode_index",
                "capture_frame_index",
                "timestamp",
                "is_frozen_frame",
                "is_emission_frame",
                "is_visual_ack_frame",
            ))
            writer.writeheader()
            for index, sample in enumerate(samples, start=1):
                writer.writerow({
                    "clip_frame_index": index,
                    "episode_index": sample.episode_index,
                    "capture_frame_index": sample.frame_index,
                    "timestamp": sample.timestamp,
                    "is_frozen_frame": (
                        self._roi_markers.get(
                            (sample.episode_index, "frozen"),
                            (None, None),
                        )[0] == sample.frame_index
                    ),
                    "is_emission_frame": (
                        self._roi_markers.get(
                            (sample.episode_index, "emission"),
                            (None, None),
                        )[0] == sample.frame_index
                    ),
                    "is_visual_ack_frame": (
                        self._roi_markers.get(
                            (sample.episode_index, "visual_ack"),
                            (None, None),
                        )[0] == sample.frame_index
                    ),
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
            "press_roi_pre_roll_seconds": self._roi_pre_roll_seconds,
            "press_roi_post_roll_seconds": self._roi_post_roll_seconds,
            "press_roi_frames_dropped": self._dropped_roi_frames,
            "press_review_items_path": str(review_path),
            "press_episode_review_items": list(self._review_items),
            "press_shadow_proposal_count": sum(
                int(row["exactly_once_proposal_count"])
                for row in self._review_items
            ),
        }

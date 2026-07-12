"""Temporal PRESS panel confirmation and variable-length sequence consensus."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from src.fishing_v2.domain.observations import PressObservation


@dataclass(frozen=True)
class PressSequenceAggregationConfig:
    panel_confirmation_frames: int = 2
    panel_geometry_tolerance: float = 0.12
    sequence_window_frames: int = 5
    sequence_consensus_frames: int = 3
    per_key_min_aggregated_confidence: float = 0.68
    sequence_min_aggregated_confidence: float = 0.68

    def __post_init__(self) -> None:
        if self.panel_confirmation_frames < 1:
            raise ValueError("panel_confirmation_frames must be positive")
        if not 0.0 <= self.panel_geometry_tolerance <= 0.5:
            raise ValueError("panel_geometry_tolerance must be within 0..0.5")
        if self.sequence_window_frames < self.sequence_consensus_frames or self.sequence_consensus_frames < 1:
            raise ValueError("sequence window must contain the required consensus frames")
        if not 0.0 <= self.per_key_min_aggregated_confidence <= 1.0:
            raise ValueError("per-key aggregated confidence must be within 0..1")
        if not 0.0 <= self.sequence_min_aggregated_confidence <= 1.0:
            raise ValueError("sequence aggregated confidence must be within 0..1")


@dataclass(frozen=True)
class PressSequenceAggregation:
    panel_confirmed: bool
    stable_panel_frames: int
    stable_key_box_count: int
    sequence_candidate: tuple[str, ...]
    sequence_ready: bool
    sequence_confidence: float
    per_key_confidence: tuple[float, ...]
    qualification_reason: str
    selected_clean_frame: int | None = None


class PressSequenceTemporalAggregator:
    """Aggregate only within one continuous structurally confirmed panel episode."""

    def __init__(self, config: PressSequenceAggregationConfig | None = None) -> None:
        self.config = config or PressSequenceAggregationConfig()
        self._panel_frames = 0
        self._missing_frames = 0
        self._window: deque[PressObservation] = deque(maxlen=self.config.sequence_window_frames)
        self._frozen_clean_sequence: tuple[str, ...] | None = None
        self._frozen_clean_confidence: tuple[float, ...] = ()
        self._frozen_clean_frame: int | None = None
        self._input_effect_seen = False

    def reset(self) -> None:
        self._panel_frames = 0
        self._missing_frames = 0
        self._window.clear()
        self._frozen_clean_sequence = None
        self._frozen_clean_confidence = ()
        self._frozen_clean_frame = None
        self._input_effect_seen = False

    @staticmethod
    def _boxes(observation: PressObservation) -> list[dict[str, Any]]:
        boxes = observation.evidence.get("key_boxes", ())
        return [dict(item) for item in boxes if isinstance(item, dict)]

    @staticmethod
    def _layout(observation: PressObservation) -> tuple[float, ...] | None:
        boxes = PressSequenceTemporalAggregator._boxes(observation)
        panel = observation.evidence.get("panel_bbox")
        if not boxes or not isinstance(panel, (list, tuple)) or len(panel) != 4:
            return None
        width = max(1.0, float(panel[2]) - float(panel[0]))
        return tuple(
            ((float(box["bbox"][0]) + float(box["bbox"][2])) * 0.5 - float(panel[0])) / width
            for box in boxes
            if isinstance(box.get("bbox"), (list, tuple)) and len(box["bbox"]) == 4
        )

    def _consistent_frames(self) -> list[PressObservation]:
        usable = [item for item in self._window if item.panel_present and item.key_box_count > 0]
        if not usable:
            return []
        count = Counter(item.key_box_count for item in usable).most_common(1)[0][0]
        same_count = [item for item in usable if item.key_box_count == count]
        reference = self._layout(same_count[-1])
        if reference is None or len(reference) != count:
            return same_count
        consistent: list[PressObservation] = []
        for item in same_count:
            layout = self._layout(item)
            if layout is not None and len(layout) == count and max(
                abs(actual - expected) for actual, expected in zip(layout, reference, strict=True)
            ) <= self.config.panel_geometry_tolerance:
                consistent.append(item)
        return consistent

    @staticmethod
    def _candidate_scores(box: dict[str, Any]) -> dict[str, float]:
        candidates = box.get("top_candidates", ())
        scores = {
            str(item.get("key", "?")): float(item.get("confidence", 0.0))
            for item in candidates
            if isinstance(item, dict) and str(item.get("key", "?")) in "WASD"
        }
        key = str(box.get("key", "?"))
        if key in "WASD":
            scores.setdefault(key, float(box.get("confidence", 0.0)))
        return scores

    def _aggregate_sequence(
        self, frames: list[PressObservation]
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if not frames:
            return (), ()
        boxes_by_frame = [self._boxes(item) for item in frames]
        count = min(len(boxes) for boxes in boxes_by_frame)
        sequence: list[str] = []
        confidences: list[float] = []
        for index in range(count):
            scores_by_frame = [self._candidate_scores(boxes[index]) for boxes in boxes_by_frame]
            totals = {
                key: sum(scores.get(key, 0.0) for scores in scores_by_frame)
                for key in "WASD"
            }
            key = max(totals, key=totals.get)
            mean_score = totals[key] / len(scores_by_frame)
            agreement = sum(
                bool(scores) and max(scores, key=scores.get) == key for scores in scores_by_frame
            ) / len(scores_by_frame)
            mean_margin = sum(
                max(0.0, scores.get(key, 0.0) - max(
                    (score for label, score in scores.items() if label != key),
                    default=0.0,
                ))
                for scores in scores_by_frame
            ) / len(scores_by_frame)
            margin_quality = min(1.0, mean_margin / 0.15)
            # Consensus is independent evidence. It raises confidence only when
            # the same unambiguous glyph wins across geometrically aligned frames.
            # Repeated but near-tied template scores remain below readiness.
            aggregated = 0.55 * mean_score + 0.25 * agreement + 0.20 * margin_quality
            sequence.append(key)
            confidences.append(max(0.0, min(1.0, aggregated)))
        return tuple(sequence), tuple(confidences)

    def update(self, observation: PressObservation) -> PressSequenceAggregation:
        if observation.panel_present:
            self._panel_frames += 1
            self._missing_frames = 0
            self._window.append(observation)
            if observation.evidence.get("input_effect_detected") is True:
                self._input_effect_seen = True
            if (
                self._frozen_clean_sequence is None
                and not self._input_effect_seen
                and observation.evidence.get("clean_frame_eligible") is True
                and observation.evidence.get("arrow_sequence_ready") is True
                and observation.sequence_candidate
            ):
                slots = observation.evidence.get("slots", ())
                occupied = [
                    item for item in slots
                    if isinstance(item, dict) and item.get("occupancy") == "OCCUPIED"
                ]
                self._frozen_clean_sequence = tuple(observation.sequence_candidate)
                self._frozen_clean_confidence = tuple(
                    float(item.get("arrow_confidence", 0.0)) for item in occupied
                )
                self._frozen_clean_frame = observation.frame_index
        else:
            self._missing_frames += 1
            if self._missing_frames >= self.config.panel_confirmation_frames:
                self.reset()
            return PressSequenceAggregation(
                False, self._panel_frames, 0, (), False, 0.0, (), "panel_not_present"
            )

        panel_confirmed = self._panel_frames >= self.config.panel_confirmation_frames
        if self._frozen_clean_sequence and not panel_confirmed:
            per_key = self._frozen_clean_confidence
            confidence = sum(per_key) / len(per_key) if per_key else observation.sequence_confidence
            return PressSequenceAggregation(
                False,
                self._panel_frames,
                len(self._frozen_clean_sequence),
                self._frozen_clean_sequence,
                False,
                round(confidence, 4),
                tuple(round(value, 4) for value in per_key),
                "panel_confirmation_pending_with_frozen_clean_sequence",
                self._frozen_clean_frame,
            )
        if panel_confirmed and self._frozen_clean_sequence:
            per_key = self._frozen_clean_confidence
            confidence = sum(per_key) / len(per_key) if per_key else observation.sequence_confidence
            return PressSequenceAggregation(
                True,
                self._panel_frames,
                len(self._frozen_clean_sequence),
                self._frozen_clean_sequence,
                True,
                round(confidence, 4),
                tuple(round(value, 4) for value in per_key),
                "earliest_clean_arrow_sequence_frozen",
                self._frozen_clean_frame,
            )
        frames = self._consistent_frames()
        candidate, per_key = self._aggregate_sequence(frames)
        stable_count = len(candidate)
        confidence = sum(per_key) / len(per_key) if per_key else 0.0
        if not panel_confirmed:
            reason = "panel_confirmation_pending"
        elif len(frames) < self.config.sequence_consensus_frames:
            reason = "sequence_consensus_pending"
        elif not candidate:
            reason = "sequence_not_recoverable_from_replay"
        elif any(score < self.config.per_key_min_aggregated_confidence for score in per_key):
            reason = "per_key_aggregated_confidence_below_threshold"
        elif confidence < self.config.sequence_min_aggregated_confidence:
            reason = "sequence_aggregated_confidence_below_threshold"
        else:
            reason = "temporal_sequence_consensus_ready"
        ready = reason == "temporal_sequence_consensus_ready"
        return PressSequenceAggregation(
            panel_confirmed,
            self._panel_frames,
            stable_count,
            candidate,
            ready,
            round(confidence, 4),
            tuple(round(value, 4) for value in per_key),
            reason,
            self._frozen_clean_frame,
        )

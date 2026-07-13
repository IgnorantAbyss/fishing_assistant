"""Non-OCR, non-neural fixed-ROI prototype Prompt observer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.fishing_v2.data.prompt_annotation import PromptAnnotationKind
from src.fishing_v2.data.prompt_roi import PromptROICandidate
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import PromptObservation, PromptObservationKind


OBSERVER_VERSION = "prototype_v1"
OPERATIONAL_LABELS = (
    "IDLE_CAST",
    "WAITING_IN_PROGRESS",
    "READY_BITE",
    "HOOK_INSTRUCTION",
    "PRESS_INSTRUCTION",
)


def validate_prompt_input(frame: np.ndarray) -> None:
    """Enforce the shared Replay/Live input contract: uint8 HxWx3 BGR contiguous."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("Prompt input must be a numpy array")
    if frame.dtype != np.uint8:
        raise ValueError(f"Prompt input dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Prompt input must be HxWx3 BGR, got shape {frame.shape}")
    if not frame.flags.c_contiguous:
        raise ValueError("Prompt input must be C-contiguous")


@dataclass(frozen=True)
class PromptPrototype:
    prototype_id: str
    label: str
    source_session: str
    source_frame: int
    feature: np.ndarray


@dataclass(frozen=True)
class PrototypePrediction:
    predicted_label: str
    similarity: float
    second_label: str
    second_similarity: float
    ambiguity_margin: float
    rejection_reason: str | None
    prototype_id: str | None
    observer_version: str = OBSERVER_VERSION


@dataclass(frozen=True)
class PrototypePromptModel:
    prototypes: tuple[PromptPrototype, ...]
    class_thresholds: Mapping[str, float]
    ambiguity_threshold: float
    training_sessions: tuple[str, ...]
    calibration_sessions: tuple[str, ...]
    idle_stability_frames: int = 1
    observer_version: str = OBSERVER_VERSION

    def raw_scores(self, feature: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return ({label: -1.0 for label in OPERATIONAL_LABELS}, {})
        vector = vector / norm
        scores = {label: -1.0 for label in OPERATIONAL_LABELS}
        prototype_ids: dict[str, str] = {}
        for prototype in self.prototypes:
            score = float(np.dot(vector, prototype.feature))
            if score > scores[prototype.label]:
                scores[prototype.label] = score
                prototype_ids[prototype.label] = prototype.prototype_id
        return scores, prototype_ids

    def predict_feature(self, feature: np.ndarray) -> PrototypePrediction:
        scores, prototype_ids = self.raw_scores(feature)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        (top_label, top_score), (second_label, second_score) = ranked[:2]
        margin = top_score - second_score
        threshold = float(self.class_thresholds.get(top_label, 1.0))
        rejection_reason = None
        predicted = top_label
        if top_score < threshold:
            predicted = PromptObservationKind.UNKNOWN.value
            rejection_reason = "low_similarity"
        elif margin < self.ambiguity_threshold:
            predicted = PromptObservationKind.UNKNOWN.value
            rejection_reason = "ambiguous_top_two"
        return PrototypePrediction(
            predicted,
            top_score,
            second_label,
            second_score,
            margin,
            rejection_reason,
            prototype_ids.get(top_label),
        )


def extract_prompt_feature(crop: np.ndarray) -> np.ndarray:
    """Extract locally normalized text/edge evidence without fitted parameters."""
    if crop is None or crop.size == 0:
        raise ValueError("Prompt ROI crop is empty")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = gray.astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4)).apply(gray)
    background = cv2.GaussianBlur(clahe, (0, 0), sigmaX=7.0, sigmaY=7.0)
    positive = cv2.subtract(clahe, background)
    bright_local = np.where((gray >= 145) & (positive >= 8), positive, 0).astype(np.uint8)
    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    bright_small = cv2.resize(bright_local, (170, 16), interpolation=cv2.INTER_AREA)
    edge_small = cv2.resize(gradient, (170, 16), interpolation=cv2.INTER_AREA)
    channels = []
    for channel in (bright_small, edge_small):
        vector = channel.astype(np.float32).reshape(-1)
        vector -= float(vector.mean())
        norm = float(np.linalg.norm(vector))
        channels.append(vector / norm if norm > 1e-8 else vector)
    feature = np.concatenate(channels).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 1e-8 else feature


def _representative_index(features: np.ndarray, indices: Sequence[int]) -> int:
    if not indices:
        raise ValueError("Cannot choose a prototype from an empty class/session group")
    sampled = np.asarray(indices, dtype=np.int64)
    if len(sampled) > 96:
        sampled = sampled[np.linspace(0, len(sampled) - 1, 96, dtype=int)]
    candidates = features[sampled]
    centroid = candidates.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-8:
        centroid /= norm
    return int(sampled[int(np.argmax(candidates @ centroid))])


def build_prototypes(
    rows: Sequence[Mapping[str, str]],
    features: np.ndarray,
    training_sessions: Sequence[str],
) -> tuple[PromptPrototype, ...]:
    training = tuple(sorted(training_sessions))
    if not training:
        raise ValueError("Prototype training requires at least one session")
    if len(rows) != len(features):
        raise ValueError("Manifest rows and feature count differ")
    prototypes: list[PromptPrototype] = []
    for label in OPERATIONAL_LABELS:
        for session_id in training:
            indices = [
                index
                for index, row in enumerate(rows)
                if row["session_id"] == session_id and row["label"] == label
            ]
            if not indices:
                continue
            selected = _representative_index(features, indices)
            row = rows[selected]
            vector = features[selected].astype(np.float32, copy=True)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                vector /= norm
            prototypes.append(PromptPrototype(
                prototype_id=f"{label}:{session_id}:{int(row['frame_index']):06d}",
                label=label,
                source_session=session_id,
                source_frame=int(row["frame_index"]),
                feature=vector,
            ))
    missing = set(OPERATIONAL_LABELS) - {item.label for item in prototypes}
    if missing:
        raise ValueError(f"Prototype training labels are missing: {sorted(missing)}")
    if any(item.label == PromptAnnotationKind.IGNORE.value for item in prototypes):
        raise AssertionError("IGNORE must never become a Prompt prototype")
    return tuple(prototypes)


def _score_matrix(prototypes: Sequence[PromptPrototype], features: np.ndarray) -> np.ndarray:
    matrix = np.full((len(features), len(OPERATIONAL_LABELS)), -1.0, dtype=np.float32)
    for label_index, label in enumerate(OPERATIONAL_LABELS):
        vectors = np.stack([item.feature for item in prototypes if item.label == label])
        matrix[:, label_index] = np.max(features @ vectors.T, axis=1)
    return matrix


def _binary_f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def calibrate_rejection(
    labels: Sequence[str],
    score_matrix: np.ndarray,
) -> tuple[dict[str, float], float]:
    if len(labels) != len(score_matrix):
        raise ValueError("Calibration label and score counts differ")
    top_indices = np.argmax(score_matrix, axis=1)
    top_scores = score_matrix[np.arange(len(score_matrix)), top_indices]
    ranked = np.sort(score_matrix, axis=1)
    margins = ranked[:, -1] - ranked[:, -2]
    label_array = np.asarray(labels)
    thresholds: dict[str, float] = {}
    for class_index, label in enumerate(OPERATIONAL_LABELS):
        relevant = top_indices == class_index
        candidates = np.unique(np.quantile(top_scores[relevant], np.linspace(0.0, 1.0, 101))) if relevant.any() else np.array([1.0])
        best = (-1.0, 1.0)
        for threshold in candidates:
            accepted = relevant & (top_scores >= threshold)
            truth = label_array == label
            tp = int(np.sum(accepted & truth))
            fp = int(np.sum(accepted & ~truth))
            fn = int(np.sum(~accepted & truth))
            score = _binary_f1(tp, fp, fn)
            key = (score, float(threshold))
            if key > best:
                best = key
        thresholds[label] = best[1]

    threshold_accept = np.asarray([
        top_scores[index] >= thresholds[OPERATIONAL_LABELS[top_indices[index]]]
        for index in range(len(labels))
    ])
    margin_candidates = np.unique(np.concatenate((np.array([0.0]), np.quantile(margins, np.linspace(0.0, 0.5, 101)))))
    best_margin = (float("-inf"), 0.0)
    for margin_threshold in margin_candidates:
        accepted = threshold_accept & (margins >= margin_threshold)
        f1_values = []
        for class_index, label in enumerate(OPERATIONAL_LABELS):
            predicted = accepted & (top_indices == class_index)
            truth = label_array == label
            f1_values.append(_binary_f1(
                int(np.sum(predicted & truth)),
                int(np.sum(predicted & ~truth)),
                int(np.sum(~predicted & truth)),
            ))
        ignore_acceptance = float(np.mean(accepted[label_array == PromptAnnotationKind.IGNORE.value])) if np.any(label_array == PromptAnnotationKind.IGNORE.value) else 0.0
        objective = float(np.mean(f1_values)) - 0.25 * ignore_acceptance
        key = (objective, float(margin_threshold))
        if key > best_margin:
            best_margin = key
    return thresholds, best_margin[1]


def fit_prototype_model(
    rows: Sequence[Mapping[str, str]],
    features: np.ndarray,
    training_sessions: Sequence[str],
) -> PrototypePromptModel:
    """Fit prototypes and rejection using nested training-session holdouts only."""
    training = tuple(sorted(training_sessions))
    final_prototypes = build_prototypes(rows, features, training)
    calibration_scores: list[np.ndarray] = []
    calibration_labels: list[str] = []
    calibration_frame_sessions: list[str] = []
    calibration_sessions: list[str] = []
    for calibration_session in training:
        inner_training = tuple(item for item in training if item != calibration_session)
        if not inner_training:
            continue
        inner_prototypes = build_prototypes(rows, features, inner_training)
        indices = [
            index for index, row in enumerate(rows) if row["session_id"] == calibration_session
        ]
        calibration_scores.append(_score_matrix(inner_prototypes, features[indices]))
        calibration_labels.extend(rows[index]["label"] for index in indices)
        calibration_frame_sessions.extend(calibration_session for _ in indices)
        calibration_sessions.append(calibration_session)
    if not calibration_scores:
        raise ValueError("At least two training sessions are required for calibration")
    thresholds, margin = calibrate_rejection(
        calibration_labels, np.concatenate(calibration_scores, axis=0)
    )
    combined_scores = np.concatenate(calibration_scores, axis=0)
    top_indices = np.argmax(combined_scores, axis=1)
    top_scores = combined_scores[np.arange(len(combined_scores)), top_indices]
    ranked_scores = np.sort(combined_scores, axis=1)
    raw_predictions = [
        OPERATIONAL_LABELS[class_index]
        if top_scores[index] >= thresholds[OPERATIONAL_LABELS[class_index]]
        and top_scores[index] - ranked_scores[index, -2] >= margin
        else PromptObservationKind.UNKNOWN.value
        for index, class_index in enumerate(top_indices)
    ]
    longest_ignore_idle_run = current = 0
    previous_session = None
    for session_id, expected, predicted in zip(
        calibration_frame_sessions, calibration_labels, raw_predictions
    ):
        if session_id != previous_session:
            current = 0
        if expected == PromptAnnotationKind.IGNORE.value and predicted == "IDLE_CAST":
            current += 1
            longest_ignore_idle_run = max(longest_ignore_idle_run, current)
        else:
            current = 0
        previous_session = session_id
    # One frame beyond the observed negative run separates the classes; one
    # additional frame is a fixed transition-sampling safety margin.  Both the
    # run length and resulting parameter are computed without the outer fold.
    idle_stability_frames = longest_ignore_idle_run + 2
    return PrototypePromptModel(
        final_prototypes,
        thresholds,
        margin,
        training,
        tuple(calibration_sessions),
        idle_stability_frames,
    )


class PrototypePromptObserver:
    def __init__(self, model: PrototypePromptModel, roi: PromptROICandidate) -> None:
        self.model = model
        self.roi = roi
        self._idle_streak = 0

    def observe(self, frame: np.ndarray, context: FrameContext) -> PromptObservation:
        validate_prompt_input(frame)
        x1, y1, x2, y2 = self.roi.pixel_bounds(frame.shape[1], frame.shape[0])
        feature = extract_prompt_feature(frame[y1:y2, x1:x2])
        prediction = self.model.predict_feature(feature)
        raw_label = prediction.predicted_label
        self._idle_streak = self._idle_streak + 1 if raw_label == "IDLE_CAST" else 0
        temporal_rejected = (
            raw_label == "IDLE_CAST" and self._idle_streak < self.model.idle_stability_frames
        )
        predicted_label = "UNKNOWN" if temporal_rejected else raw_label
        kind = PromptObservationKind(predicted_label)
        scores, _ = self.model.raw_scores(feature)
        confidence = 0.0 if temporal_rejected else max(0.0, min(1.0, (prediction.similarity + 1.0) / 2.0))
        return PromptObservation(
            kind=kind,
            confidence=confidence,
            probabilities=scores,
            source=OBSERVER_VERSION,
            frame_index=context.frame_index,
            timestamp=context.timestamp,
            evidence={
                "predicted_label": predicted_label,
                "raw_predicted_label": raw_label,
                "similarity": prediction.similarity,
                "second_label": prediction.second_label,
                "second_similarity": prediction.second_similarity,
                "ambiguity_margin": prediction.ambiguity_margin,
                "rejection_reason": "idle_temporal_guard" if temporal_rejected else prediction.rejection_reason,
                "prototype_id": prediction.prototype_id,
                "observer_version": prediction.observer_version,
                "approved_roi": list(self.roi.pixel),
                "idle_streak": self._idle_streak,
                "idle_stability_frames": self.model.idle_stability_frames,
                "input_contract": "uint8_hwc3_bgr_contiguous_0_255",
                "feature_sha256": hashlib.sha256(feature.tobytes()).hexdigest(),
            },
        )

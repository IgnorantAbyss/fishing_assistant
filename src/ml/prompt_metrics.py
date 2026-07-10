"""Dependency-light metrics and validation-only confidence selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from src.ml.prompt_dataset import CLASS_NAMES


UNKNOWN = "UNKNOWN"


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], class_names: Sequence[str] = CLASS_NAMES
) -> list[list[int]]:
    lookup = {name: index for index, name in enumerate(class_names)}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth not in lookup:
            raise ValueError(f"Unknown ground-truth class: {truth}")
        if prediction in lookup:
            matrix[lookup[truth], lookup[prediction]] += 1
    return matrix.tolist()


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], class_names: Sequence[str] = CLASS_NAMES
) -> dict[str, object]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metrics require equally sized, non-empty truth and prediction lists")
    matrix = np.asarray(confusion_matrix(y_true, y_pred, class_names), dtype=np.int64)
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = sum(truth == name for truth in y_true)
        predicted = sum(prediction == name for prediction in y_pred)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    correct = sum(truth == prediction for truth, prediction in zip(y_true, y_pred, strict=True))
    idle = class_names.index("IDLE")
    waiting = class_names.index("WAITING")
    idle_support = int(matrix[idle, :].sum()) + sum(
        truth == "IDLE" and prediction == UNKNOWN
        for truth, prediction in zip(y_true, y_pred, strict=True)
    )
    waiting_support = int(matrix[waiting, :].sum()) + sum(
        truth == "WAITING" and prediction == UNKNOWN
        for truth, prediction in zip(y_true, y_pred, strict=True)
    )
    idle_to_waiting = int(matrix[idle, waiting])
    waiting_to_idle = int(matrix[waiting, idle])
    mutual_denominator = idle_support + waiting_support
    return {
        "sample_count": len(y_true),
        "accuracy": correct / len(y_true),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "idle_to_waiting": idle_to_waiting,
        "waiting_to_idle": waiting_to_idle,
        "idle_waiting_mutual_confusion_rate": (
            (idle_to_waiting + waiting_to_idle) / mutual_denominator
            if mutual_denominator
            else 0.0
        ),
        "unknown_ratio": sum(prediction == UNKNOWN for prediction in y_pred) / len(y_pred),
    }


def probabilities_to_predictions(
    probabilities: Sequence[Sequence[float]], threshold: float
) -> tuple[list[str], list[float]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError("Probabilities must have shape [samples, 4]")
    indices = values.argmax(axis=1)
    confidence = values.max(axis=1)
    predictions = [
        CLASS_NAMES[int(index)] if score >= threshold else UNKNOWN
        for index, score in zip(indices, confidence, strict=True)
    ]
    return predictions, confidence.tolist()


def select_confidence_threshold(
    probabilities: Sequence[Sequence[float]],
    y_true: Sequence[str],
    candidates: Iterable[float],
    max_unknown_ratio: float,
    *,
    source_split: str,
) -> dict[str, object]:
    if source_split != "validation":
        raise ValueError("Confidence threshold search is permitted on validation only")
    ordered = sorted({float(value) for value in candidates})
    if not ordered:
        raise ValueError("At least one threshold candidate is required")
    candidate_metrics: list[dict[str, float]] = []
    for threshold in ordered:
        predictions, _ = probabilities_to_predictions(probabilities, threshold)
        metrics = classification_metrics(y_true, predictions)
        candidate_metrics.append(
            {
                "threshold": threshold,
                "unknown_ratio": float(metrics["unknown_ratio"]),
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
    eligible = [
        item for item in candidate_metrics if item["unknown_ratio"] <= max_unknown_ratio
    ]
    warning = False
    if eligible:
        selected = max(eligible, key=lambda item: item["threshold"])
        rule = "highest_threshold_within_validation_unknown_limit"
    else:
        selected = min(
            candidate_metrics,
            key=lambda item: (abs(item["unknown_ratio"] - max_unknown_ratio), -item["threshold"]),
        )
        warning = True
        rule = "closest_validation_unknown_ratio_to_limit"
    return {
        "selected_threshold": selected["threshold"],
        "validation_unknown_ratio": selected["unknown_ratio"],
        "selection_rule": rule,
        "confidence_gate_warning": warning,
        "candidate_metrics": candidate_metrics,
    }


def metrics_by_session(
    y_true: Sequence[str], y_pred: Sequence[str], sessions: Sequence[str]
) -> dict[str, dict[str, object]]:
    if not (len(y_true) == len(y_pred) == len(sessions)):
        raise ValueError("Session metrics inputs must have equal lengths")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, session_id in enumerate(sessions):
        grouped[session_id].append(index)
    return {
        session_id: classification_metrics(
            [y_true[index] for index in indices],
            [y_pred[index] for index in indices],
        )
        for session_id, indices in sorted(grouped.items())
    }


def confidence_distribution(confidences: Sequence[float]) -> dict[str, float]:
    values = np.asarray(confidences, dtype=np.float64)
    if values.size == 0:
        return {name: 0.0 for name in ("min", "mean", "p50", "p95", "max")}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def latency_distribution(latencies_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }

"""Session-isolated evaluation for the prototype Prompt observer."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from src.fishing_v2.data.prompt_annotation import PromptAnnotationKind
from src.fishing_v2.data.prompt_observation_dataset import (
    FORMAL_SESSION_IDS,
    read_prompt_observation_manifest,
    resolve_roi_path,
)
from src.fishing_v2.perception.prototype_prompt_observer import (
    OPERATIONAL_LABELS,
    PrototypePrediction,
    fit_prototype_model,
    extract_prompt_feature,
)


def load_or_build_features(
    rows: Sequence[Mapping[str, str]],
    dataset_root: str | Path,
    *,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    dataset_root = Path(dataset_root)
    cache = Path(cache_path) if cache_path else None
    manifest_signature = hashlib.sha256(
        "\n".join(row["image_sha256"] for row in rows).encode("ascii")
    ).hexdigest()
    if cache and cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if str(data["manifest_signature"]) == manifest_signature:
            return data["features"].astype(np.float32, copy=False)
    features: list[np.ndarray] = []
    for row in rows:
        path = resolve_roi_path(row, dataset_root)
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Prompt ROI hash mismatch: {path}")
        crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError(f"Could not decode Prompt ROI: {path}")
        features.append(extract_prompt_feature(crop))
    matrix = np.stack(features).astype(np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=matrix, manifest_signature=np.asarray(manifest_signature))
    return matrix


def _prediction_dict(prediction: PrototypePrediction) -> dict[str, Any]:
    return {
        "predicted_label": prediction.predicted_label,
        "similarity": prediction.similarity,
        "second_label": prediction.second_label,
        "second_similarity": prediction.second_similarity,
        "ambiguity_margin": prediction.ambiguity_margin,
        "rejection_reason": prediction.rejection_reason,
        "prototype_id": prediction.prototype_id,
        "observer_version": prediction.observer_version,
    }


def _class_metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in OPERATIONAL_LABELS:
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(item == label for item in expected),
        }
    return result


def _longest_run(items: Sequence[str], target: str) -> int:
    longest = current = 0
    for item in items:
        current = current + 1 if item == target else 0
        longest = max(longest, current)
    return longest


def _segment_events(
    predictions: Mapping[tuple[str, int], Mapping[str, Any]],
    session_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for session_id in FORMAL_SESSION_IDS:
        raw = yaml.safe_load(
            (session_root / session_id / "prompt_ground_truth.yaml").read_text(encoding="utf-8")
        )
        for segment_index, segment in enumerate(raw["segments"], start=1):
            label = str(segment["observation"])
            if label == PromptAnnotationKind.IGNORE.value:
                continue
            start, end = int(segment["start"]), int(segment["end"])
            segment_predictions = [
                str(predictions[(session_id, frame)]["predicted_label"])
                for frame in range(start, end + 1)
            ]
            correct_offsets = [
                offset for offset, item in enumerate(segment_predictions) if item == label
            ]
            first_correct = start + correct_offsets[0] if correct_offsets else None
            wrong_valid = sum(
                item not in {label, "UNKNOWN"} for item in segment_predictions
            )
            next_operational_label = next((
                str(item["observation"])
                for item in raw["segments"][segment_index:]
                if str(item["observation"]) != PromptAnnotationKind.IGNORE.value
            ), None)
            events.append({
                "session_id": session_id,
                "segment_index": segment_index,
                "label": label,
                "ground_truth_start": start,
                "ground_truth_end": end,
                "first_correct_predicted_frame": first_correct,
                "detection_latency_frames": first_correct - start if first_correct is not None else None,
                "detected_at_least_once": bool(correct_offsets),
                "longest_unknown_run": _longest_run(segment_predictions, "UNKNOWN"),
                "wrong_operational_prediction_frames": wrong_valid,
                "next_operational_label": next_operational_label,
                "premature_next_prompt_prediction_frames": (
                    sum(item == next_operational_label for item in segment_predictions)
                    if next_operational_label else 0
                ),
                "only_rejected_as_unknown": not correct_offsets and wrong_valid == 0,
            })
    by_label: dict[str, Any] = {}
    for label in OPERATIONAL_LABELS:
        label_events = [event for event in events if event["label"] == label]
        latencies = [
            int(event["detection_latency_frames"])
            for event in label_events
            if event["detection_latency_frames"] is not None
        ]
        by_label[label] = {
            "episodes": len(label_events),
            "detected_episodes": sum(bool(event["detected_at_least_once"]) for event in label_events),
            "episode_recall": (
                sum(bool(event["detected_at_least_once"]) for event in label_events) / len(label_events)
                if label_events else 0.0
            ),
            "mean_first_detection_latency_frames": float(np.mean(latencies)) if latencies else None,
            "max_first_detection_latency_frames": max(latencies) if latencies else None,
        }
    return events, by_label


def evaluate_loso(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    session_root: str | Path,
    feature_cache: str | Path | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    rows = read_prompt_observation_manifest(manifest_path)
    sessions = tuple(sorted({row["session_id"] for row in rows}))
    if sessions != tuple(sorted(FORMAL_SESSION_IDS)):
        raise ValueError("LOSO manifest must contain exactly the seven formal sessions")
    features = load_or_build_features(rows, dataset_root, cache_path=feature_cache)
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    fold_models: dict[str, Any] = {}
    folds: list[dict[str, Any]] = []
    for held_out in FORMAL_SESSION_IDS:
        training = tuple(session for session in FORMAL_SESSION_IDS if session != held_out)
        model = fit_prototype_model(rows, features, training)
        if held_out in model.training_sessions or held_out in model.calibration_sessions:
            raise AssertionError("Held-out session leaked into prototype or threshold calibration")
        indices = [index for index, row in enumerate(rows) if row["session_id"] == held_out]
        fold_expected: list[str] = []
        fold_predicted: list[str] = []
        idle_streak = 0
        for index in indices:
            prediction = model.predict_feature(features[index])
            row = rows[index]
            raw_label = prediction.predicted_label
            idle_streak = idle_streak + 1 if raw_label == "IDLE_CAST" else 0
            temporal_rejected = raw_label == "IDLE_CAST" and idle_streak < model.idle_stability_frames
            prediction_data = _prediction_dict(prediction)
            prediction_data["raw_predicted_label"] = raw_label
            prediction_data["idle_streak"] = idle_streak
            prediction_data["idle_stability_frames"] = model.idle_stability_frames
            if temporal_rejected:
                prediction_data["predicted_label"] = "UNKNOWN"
                prediction_data["rejection_reason"] = "idle_temporal_guard"
            predictions[(held_out, int(row["frame_index"]))] = prediction_data
            if row["label"] != PromptAnnotationKind.IGNORE.value:
                fold_expected.append(row["label"])
                fold_predicted.append(str(prediction_data["predicted_label"]))
        metrics = _class_metrics(fold_expected, fold_predicted)
        folds.append({
            "held_out_session": held_out,
            "training_sessions": list(model.training_sessions),
            "calibration_sessions": list(model.calibration_sessions),
            "prototype_count": len(model.prototypes),
            "prototype_counts": dict(Counter(item.label for item in model.prototypes)),
            "class_thresholds": dict(model.class_thresholds),
            "ambiguity_margin": model.ambiguity_threshold,
            "idle_stability_frames": model.idle_stability_frames,
            "macro_f1": float(np.mean([item["f1"] for item in metrics.values()])),
            "unknown_prediction_rate": sum(item == "UNKNOWN" for item in fold_predicted) / len(fold_predicted),
        })
        fold_models[held_out] = model

    expected = []
    predicted = []
    ignore_predictions = []
    session_metrics: dict[str, Any] = {}
    session_label_counts: dict[str, dict[str, int]] = {}
    for session_id in FORMAL_SESSION_IDS:
        session_expected = []
        session_predicted = []
        for row in rows:
            if row["session_id"] != session_id:
                continue
            actual = row["label"]
            guess = predictions[(session_id, int(row["frame_index"]))]["predicted_label"]
            if actual == PromptAnnotationKind.IGNORE.value:
                ignore_predictions.append(guess)
            else:
                expected.append(actual)
                predicted.append(guess)
                session_expected.append(actual)
                session_predicted.append(guess)
        counts = Counter(row["label"] for row in rows if row["session_id"] == session_id)
        session_label_counts[session_id] = {
            label: counts.get(label, 0)
            for label in (*OPERATIONAL_LABELS, PromptAnnotationKind.IGNORE.value)
        }
        class_result = _class_metrics(session_expected, session_predicted)
        session_metrics[session_id] = {
            "accuracy": sum(a == b for a, b in zip(session_expected, session_predicted)) / len(session_expected),
            "macro_f1": float(np.mean([item["f1"] for item in class_result.values()])),
            "unknown_prediction_rate": sum(item == "UNKNOWN" for item in session_predicted) / len(session_predicted),
        }
    metrics = _class_metrics(expected, predicted)
    columns = (*OPERATIONAL_LABELS, "UNKNOWN")
    confusion = {
        actual: {guess: sum(a == actual and b == guess for a, b in zip(expected, predicted)) for guess in columns}
        for actual in OPERATIONAL_LABELS
    }
    events, event_metrics = _segment_events(predictions, Path(session_root))
    errors = []
    for row in rows:
        frame = int(row["frame_index"])
        prediction = predictions[(row["session_id"], frame)]
        if row["label"] != PromptAnnotationKind.IGNORE.value and prediction["predicted_label"] != row["label"]:
            errors.append({
                "session_id": row["session_id"],
                "frame_index": frame,
                "expected": row["label"],
                **prediction,
                "failure_category": (
                    "C_rejection_threshold" if prediction["rejection_reason"] == "low_similarity"
                    else "D_top1_top2_ambiguity" if prediction["rejection_reason"] == "ambiguous_top_two"
                    else "E_transition_idle_stability" if prediction["rejection_reason"] == "idle_temporal_guard"
                    else "B_prototype_coverage"
                ),
            })
    ignore_count = len(ignore_predictions)
    summary = {
        "observer_version": "prototype_v1",
        "evaluation": "seven-fold leave-one-session-out",
        "frame_count": len(rows),
        "evaluated_non_ignore_frames": len(expected),
        "ignore_frames": ignore_count,
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "session_label_counts": session_label_counts,
        "folds": folds,
        "confusion_matrix": confusion,
        "per_class": metrics,
        "macro_f1": float(np.mean([item["f1"] for item in metrics.values()])),
        "accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
        "unknown_prediction_rate": sum(item == "UNKNOWN" for item in predicted) / len(predicted),
        "ignore": {
            "unknown_rejection_rate": ignore_predictions.count("UNKNOWN") / ignore_count if ignore_count else 0.0,
            "false_acceptance": {
                label: ignore_predictions.count(label) for label in OPERATIONAL_LABELS
            },
            "false_acceptance_rate": {
                label: ignore_predictions.count(label) / ignore_count if ignore_count else 0.0
                for label in OPERATIONAL_LABELS
            },
        },
        "session_metrics": session_metrics,
        "event_metrics": event_metrics,
        "events": events,
        "representative_errors": errors[:40],
        "failure_counts": dict(Counter(item["failure_category"] for item in errors)),
    }
    return summary, predictions, fold_models


def write_prompt_observer_reports(
    summary: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Prototype PromptObserver LOSO Summary",
        "",
        f"- Frames: **{summary['frame_count']}** (evaluated {summary['evaluated_non_ignore_frames']}, IGNORE {summary['ignore_frames']})",
        f"- Macro F1: **{summary['macro_f1']:.4f}**",
        f"- Accuracy: **{summary['accuracy']:.4f}**",
        f"- Non-IGNORE UNKNOWN rate: **{summary['unknown_prediction_rate']:.4f}**",
        f"- IGNORE rejection rate: **{summary['ignore']['unknown_rejection_rate']:.4f}**",
        f"- IGNORE -> IDLE_CAST: **{summary['ignore']['false_acceptance']['IDLE_CAST']}**",
        "- Feature: per-frame CLAHE/local background subtraction plus luminance text mask and Sobel edges; cosine prototype similarity.",
        "- Isolation: each fold uses six sessions; nested training-session holdouts calibrate similarity and ambiguity rejection.",
        "- IDLE transition guard: fold-local training IGNORE runs determine the required consecutive IDLE observations.",
        "- IGNORE is calibration/evaluation negative evidence and never a prototype.",
        "",
        "## Per-class metrics",
        "",
        "| label | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in summary["per_class"].items():
        lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
    lines.extend(["", "## Confusion matrix", ""])
    columns = (*OPERATIONAL_LABELS, "UNKNOWN")
    lines.extend([
        "| actual \\ predicted | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ])
    for actual in OPERATIONAL_LABELS:
        lines.append(
            f"| {actual} | "
            + " | ".join(str(summary["confusion_matrix"][actual][guess]) for guess in columns)
            + " |"
        )
    lines.extend(["", "## Dataset counts by session", ""])
    for session_id, counts in summary["session_label_counts"].items():
        lines.append(f"- {session_id}: `{counts}`")
    lines.extend(["", "## Fold calibration", "", "| held out | prototypes | macro F1 | margin | IDLE stability | class thresholds |", "|---|---:|---:|---:|---:|---|"])
    for fold in summary["folds"]:
        thresholds = ", ".join(f"{label}={value:.4f}" for label, value in fold["class_thresholds"].items())
        lines.append(
            f"| {fold['held_out_session']} | {fold['prototype_count']} | {fold['macro_f1']:.4f} | "
            f"{fold['ambiguity_margin']:.4f} | {fold['idle_stability_frames']} | {thresholds} |"
        )
    lines.extend(["", "## Event metrics", ""])
    for label, item in summary["event_metrics"].items():
        lines.append(
            f"- {label}: {item['detected_episodes']}/{item['episodes']} episodes; "
            f"mean/max latency={item['mean_first_detection_latency_frames']}/{item['max_first_detection_latency_frames']} frames"
        )
    lines.extend(["", "## Failure classification", "", f"- {summary['failure_counts']}"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

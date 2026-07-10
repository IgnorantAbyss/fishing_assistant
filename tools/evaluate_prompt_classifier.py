"""Evaluate PromptClassifier v1 without test-time model or threshold selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.prompt_dataset import CLASS_NAMES, PromptManifestDataset  # noqa: E402
from src.ml.prompt_metrics import (  # noqa: E402
    classification_metrics,
    confidence_distribution,
    latency_distribution,
    metrics_by_session,
    probabilities_to_predictions,
    select_confidence_threshold,
)
from src.ml.prompt_model import load_prompt_checkpoint  # noqa: E402
from src.ml.prompt_transforms import build_prompt_transform  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "datasets" / "fishing_v2"
MANIFEST_PATH = DATASET_ROOT / "prompt" / "manifest.csv"
SPLIT_PATH = PROJECT_ROOT / "config" / "session_split.yaml"
CONFIG_PATH = PROJECT_ROOT / "config" / "prompt_classifier.yaml"
MODEL_DIR = PROJECT_ROOT / "models" / "prompt_classifier_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_classifier_v1"
ERROR_FIELDS = (
    "session_id",
    "frame_index",
    "ground_truth",
    "prediction",
    "confidence",
    "probabilities",
    "source_frame",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Prompt classifier config root must be a mapping")
    return data


def _collect_probabilities(
    split: str, device: torch.device
) -> tuple[list[list[float]], list[str], list[str], list[dict[str, Any]], list[float]]:
    model, checkpoint = load_prompt_checkpoint(MODEL_DIR / "best_model.pt", device=device)
    transform = build_prompt_transform(
        False,
        width=int(checkpoint.get("input_width", 320)),
        height=int(checkpoint.get("input_height", 96)),
    )
    dataset = PromptManifestDataset(MANIFEST_PATH, DATASET_ROOT, split, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    probabilities: list[list[float]] = []
    truth: list[str] = []
    sessions: list[str] = []
    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, labels, metadata in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            per_image_latency = elapsed / len(labels)
            for index, label_index in enumerate(labels.tolist()):
                values = batch_probabilities[index].tolist()
                probabilities.append(values)
                truth.append(CLASS_NAMES[label_index])
                session_id = str(metadata["session_id"][index])
                sessions.append(session_id)
                latencies_ms.append(per_image_latency)
                records.append(
                    {
                        "session_id": session_id,
                        "frame_index": int(metadata["frame_index"][index]),
                        "ground_truth": CLASS_NAMES[label_index],
                        "source_frame": str(metadata["source_frame"][index]),
                        "crop_path": str(metadata["crop_path"][index]),
                        "probabilities": values,
                    }
                )
    return probabilities, truth, sessions, records, latencies_ms


def _draw_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    cell = 125
    margin = 125
    size = margin + cell * len(CLASS_NAMES)
    image = np.full((size, size, 3), 250, dtype=np.uint8)
    maximum = max(max(row) for row in matrix) or 1
    for row, truth in enumerate(CLASS_NAMES):
        cv2.putText(image, truth, (5, margin + row * cell + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        cv2.putText(image, truth, (margin + row * cell + 15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        for column in range(len(CLASS_NAMES)):
            value = matrix[row][column]
            shade = 255 - round(180 * value / maximum)
            top, left = margin + row * cell, margin + column * cell
            cv2.rectangle(image, (left, top), (left + cell, top + cell), (shade, 255, shade), -1)
            cv2.rectangle(image, (left, top), (left + cell, top + cell), (100, 100, 100), 1)
            cv2.putText(image, str(value), (left + 42, top + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)


def _export_errors(
    prefix: str,
    records: list[dict[str, Any]],
    predictions: list[str],
    confidences: list[float],
) -> int:
    csv_path = REPORT_DIR / f"{prefix}_errors.csv"
    image_dir = REPORT_DIR / f"{prefix}_errors"
    image_dir.mkdir(parents=True, exist_ok=True)
    pair_counts: Counter[tuple[str, str]] = Counter()
    error_count = 0
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        for record, prediction, confidence in zip(records, predictions, confidences, strict=True):
            if prediction == record["ground_truth"]:
                continue
            error_count += 1
            probabilities = {
                name: record["probabilities"][index] for index, name in enumerate(CLASS_NAMES)
            }
            writer.writerow(
                {
                    "session_id": record["session_id"],
                    "frame_index": record["frame_index"],
                    "ground_truth": record["ground_truth"],
                    "prediction": prediction,
                    "confidence": f"{confidence:.8f}",
                    "probabilities": json.dumps(probabilities, separators=(",", ":")),
                    "source_frame": record["source_frame"],
                }
            )
            pair = (record["ground_truth"], prediction)
            if pair_counts[pair] >= 50:
                continue
            pair_counts[pair] += 1
            image = cv2.imread(record["crop_path"], cv2.IMREAD_COLOR)
            if image is None:
                continue
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], min(34, overlay.shape[0])), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                f"GT={record['ground_truth']} pred={prediction} conf={confidence:.3f}",
                (5, min(25, overlay.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            filename = (
                f"{record['ground_truth']}_to_{prediction}_{record['session_id']}_"
                f"{record['frame_index']:06d}.jpg"
            )
            cv2.imwrite(str(image_dir / filename), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return error_count


def _candidate_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        ("macro_f1", float(metrics["macro_f1"]), 0.85, ">="),
        ("IDLE recall", float(metrics["per_class"]["IDLE"]["recall"]), 0.85, ">="),
        ("WAITING recall", float(metrics["per_class"]["WAITING"]["recall"]), 0.90, ">="),
        ("READY recall", float(metrics["per_class"]["READY"]["recall"]), 0.85, ">="),
        (
            "IDLE/WAITING mutual confusion",
            float(metrics["idle_waiting_mutual_confusion_rate"]),
            0.08,
            "<=",
        ),
        ("UNKNOWN ratio", float(metrics["unknown_ratio"]), 0.12, "<="),
    )
    for name, actual, target, operator in checks:
        passed = actual >= target if operator == ">=" else actual <= target
        if not passed:
            failures.append(f"{name} {actual:.6f} does not satisfy {operator} {target:.2f}")
    for session_id, session_metrics in metrics["per_session"].items():
        if float(session_metrics["macro_f1"]) < 0.75:
            failures.append(f"{session_id} macro-F1 is below 0.75")
        if float(session_metrics["per_class"]["READY"]["recall"]) < 0.75:
            failures.append(f"{session_id} READY recall is below 0.75")
    return not failures, failures


def _failure_analysis(metrics: dict[str, Any]) -> dict[str, Any]:
    matrix = metrics["confusion_matrix"]
    pairs = sorted(
        (
            {
                "ground_truth": CLASS_NAMES[row],
                "prediction": CLASS_NAMES[column],
                "count": int(matrix[row][column]),
            }
            for row in range(len(CLASS_NAMES))
            for column in range(len(CLASS_NAMES))
            if row != column and matrix[row][column]
        ),
        key=lambda item: (-item["count"], item["ground_truth"], item["prediction"]),
    )
    affected_sessions = [
        session_id
        for session_id, item in metrics["per_session"].items()
        if float(item["macro_f1"]) < 0.75
        or float(item["per_class"]["READY"]["recall"]) < 0.75
    ]
    return {
        "dominant_confusion_pairs": pairs[:5],
        "data_type": "Natural held-out prompt crops; all test rows including evaluation_only samples.",
        "affected_sessions": affected_sessions,
        "recommendation": (
            "Investigate cross-session prompt appearance shift with future training-only data, "
            "prioritizing IDLE/NONE versus READY hard examples. Do not tune on this final test."
        ),
    }


def _write_report(prefix: str, split: str, metrics: dict[str, Any]) -> None:
    lines = [
        f"# PromptClassifier v1 {split.title()} Report",
        "",
        f"- Samples: {metrics['sample_count']}",
        f"- Threshold: {metrics['confidence_threshold']:.2f}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- Macro precision / recall / F1: {metrics['macro_precision']:.6f} / {metrics['macro_recall']:.6f} / {metrics['macro_f1']:.6f}",
        f"- IDLE -> WAITING: {metrics['idle_to_waiting']}",
        f"- WAITING -> IDLE: {metrics['waiting_to_idle']}",
        f"- IDLE/WAITING mutual confusion: {metrics['idle_waiting_mutual_confusion_rate']:.6f}",
        f"- UNKNOWN ratio: {metrics['unknown_ratio']:.6f}",
        f"- Mean / p50 / p95 latency (ms): {metrics['latency']['mean_ms']:.4f} / {metrics['latency']['p50_ms']:.4f} / {metrics['latency']['p95_ms']:.4f}",
        f"- Error count: {metrics['error_count']}",
        "",
        "## Per class",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        item = metrics["per_class"][name]
        lines.append(
            f"| {name} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} | {item['support']} |"
        )
    lines.extend(["", "## Per session", ""])
    for session_id, item in metrics["per_session"].items():
        lines.append(
            f"- {session_id}: macro-F1={item['macro_f1']:.6f}, "
            f"IDLE={item['per_class']['IDLE']['recall']:.6f}, "
            f"WAITING={item['per_class']['WAITING']['recall']:.6f}, "
            f"READY={item['per_class']['READY']['recall']:.6f}, "
            f"UNKNOWN={item['unknown_ratio']:.6f}"
        )
    if split == "test":
        lines.extend(
            [
                "",
                "## Integration gate",
                "",
                f"- candidate_for_integration: {str(metrics['candidate_for_integration']).lower()}",
                *(
                    [f"- Failure: {item}" for item in metrics["gate_failures"]]
                    or ["- All gates passed"]
                ),
                f"- Dominant confusions: `{metrics['failure_analysis']['dominant_confusion_pairs']}`",
                f"- Data type: {metrics['failure_analysis']['data_type']}",
                f"- Affected sessions: `{metrics['failure_analysis']['affected_sessions']}`",
                f"- Recommendation: {metrics['failure_analysis']['recommendation']}",
                "- No automatic retraining or production integration was performed.",
            ]
        )
    (REPORT_DIR / f"{prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(split: str, *, final_test: bool = False) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Only validation and test evaluation are supported")
    if split == "test" and not final_test:
        raise ValueError("Test evaluation requires --final-test")
    if split == "validation" and final_test:
        raise ValueError("--final-test is only valid with --split test")
    if not (MODEL_DIR / "best_model.pt").is_file():
        raise FileNotFoundError("best_model.pt does not exist; train before evaluation")
    config = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities, truth, sessions, records, latencies = _collect_probabilities(split, device)
    threshold_path = MODEL_DIR / "confidence_threshold.json"
    threshold_selection: dict[str, Any] | None = None
    if split == "validation":
        confidence = config["confidence"]
        threshold_selection = select_confidence_threshold(
            probabilities,
            truth,
            confidence["threshold_candidates"],
            float(confidence["max_validation_unknown_ratio"]),
            source_split="validation",
        )
        threshold_path.write_text(
            json.dumps(threshold_selection, indent=2) + "\n", encoding="utf-8"
        )
        threshold = float(threshold_selection["selected_threshold"])
        prefix = "validation"
    else:
        if not threshold_path.is_file():
            raise FileNotFoundError("Validation confidence_threshold.json must be fixed first")
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(threshold_data["selected_threshold"])
        prefix = "final_test"
    predictions, confidences = probabilities_to_predictions(probabilities, threshold)
    metrics = classification_metrics(truth, predictions)
    metrics["per_session"] = metrics_by_session(truth, predictions, sessions)
    metrics["confidence_threshold"] = threshold
    metrics["confidence_distribution"] = confidence_distribution(confidences)
    metrics["latency"] = latency_distribution(latencies)
    metrics["device"] = str(device)
    metrics["split"] = split
    metrics["threshold_selection"] = threshold_selection
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics["error_count"] = _export_errors(prefix, records, predictions, confidences)
    _draw_confusion_matrix(
        metrics["confusion_matrix"], REPORT_DIR / f"{prefix}_confusion_matrix.png"
    )
    if split == "test":
        candidate, gate_failures = _candidate_gate(metrics)
        metrics["candidate_for_integration"] = candidate
        metrics["gate_failures"] = gate_failures
        metrics["failure_analysis"] = _failure_analysis(metrics)
        run_path = REPORT_DIR / "final_test_run.json"
        previous_count = 0
        if run_path.is_file():
            try:
                previous_count = int(json.loads(run_path.read_text(encoding="utf-8")).get("run_count", 0))
            except (OSError, ValueError, TypeError):
                previous_count = 0
        run_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_sha256": _sha256(MODEL_DIR / "best_model.pt"),
            "threshold": threshold,
            "dataset_manifest_sha256": _sha256(MANIFEST_PATH),
            "session_split_sha256": _sha256(SPLIT_PATH),
            "run_count": previous_count + 1,
            "repeat_notice": (
                "Final test has been evaluated more than once; inspect audit history."
                if previous_count
                else "First recorded final-test evaluation."
            ),
        }
        run_path.write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(prefix, split, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PromptClassifier v1.")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = evaluate(args.split, final_test=args.final_test)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

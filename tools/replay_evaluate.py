"""Diagnostic evaluation of every replay frame against session ground truth.

This tool is deliberately offline.  It never captures the screen or performs
keyboard input.  Its first-pass output is raw detector evidence; smoothing and
detector calibration are separate follow-up work.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import ROI_NAMES, normalized_to_pixel_roi  # noqa: E402
from src.detectors.hook_detector import detect_hook_bar  # noqa: E402
from src.detectors.press_detector import detect_press_sequence  # noqa: E402
from src.detectors.get_detector import detect_get_window  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402
from src.replay_ground_truth import (  # noqa: E402
    EVALUATED_STATES,
    IGNORE_STATE,
    load_annotations,
    load_ground_truth,
)
from src.state_smoother import StateSmoother  # noqa: E402
from src.state_detector import StateDetector  # noqa: E402


PREDICTED_STATES = (*EVALUATED_STATES, "UNKNOWN")


@dataclass(frozen=True)
class EvaluationSummary:
    session_path: Path
    results_path: Path
    report_path: Path
    total_frames: int
    evaluated_frames: int
    ignored_frames: int
    raw_accuracy: float
    smoothed_accuracy: float
    unknown_ratio: float
    state_recalls: dict[str, float | None]
    hook_detected_count: int
    press_detected_count: int
    get_detected_count: int
    hook_bar_range: tuple[int, int] | None
    press_range: tuple[int, int] | None


def _load_ground_truth(path: Path, frame_count: int) -> dict[int, str]:
    return load_ground_truth(path, frame_count)


def _frame_range_for_state(labels: dict[int, str], state: str) -> tuple[int, int] | None:
    frames = [frame_index for frame_index, expected in labels.items() if expected == state]
    return (min(frames), max(frames)) if frames else None


def _event_range(
    annotations: dict[str, Any], name: str, fallback: tuple[int, int] | None
) -> tuple[int, int] | None:
    event = annotations.get("events", {}).get(name)
    return (int(event["start"]), int(event["end"])) if isinstance(event, dict) else fallback


def _in_range(frame_index: int, frame_range: tuple[int, int] | None) -> bool:
    return frame_range is not None and frame_range[0] <= frame_index <= frame_range[1]


def _get_window_probe(frame: np.ndarray, detector: StateDetector) -> tuple[bool, float]:
    """Independently compare only get_window ROI to GET references for diagnostics."""
    get_references = [reference for reference in detector.references if reference.state == "GET"]
    roi = detector.roi_config.rois["get_window"]
    scores = [
        detector._roi_similarity(detector._crop(frame, roi), detector._crop(reference.image, roi))
        for reference in get_references
    ]
    confidence = max(scores, default=0.0)
    return confidence >= detector.thresholds.min_confidence_for("GET"), confidence


def _hard_example_category(expected: str, detected: str) -> str | None:
    if expected == IGNORE_STATE:
        return None
    if detected == "UNKNOWN":
        return f"expected_{expected.lower()}_but_unknown"
    if expected != detected and detected in {"GET", "PRESS", "READY", "HOOK"}:
        return f"false_{detected.lower()}"
    return None


def _annotate_frame(frame: np.ndarray, detector: StateDetector, row: dict[str, Any]) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    palette = ((0, 215, 255), (0, 220, 0), (0, 140, 255), (230, 80, 230), (255, 255, 255), (255, 200, 0))
    for index, roi_name in enumerate(ROI_NAMES):
        left, top, right, bottom = normalized_to_pixel_roi(detector.roi_config.rois[roi_name], width, height)
        colour = palette[index % len(palette)]
        cv2.rectangle(annotated, (left, top), (right, bottom), colour, 2)
        cv2.putText(annotated, roi_name, (left, max(22, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    header = f"#{row['frame_index']} expected={row['expected_state']} detected={row['detected_state']} conf={float(row['confidence']):.2f}"
    cv2.rectangle(annotated, (8, 8), (min(width - 8, 860), 46), (0, 0, 0), -1)
    cv2.putText(annotated, header, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return annotated


def _save_roi_montage(frame: np.ndarray, detector: StateDetector, destination: Path) -> None:
    height, width = frame.shape[:2]
    crops = []
    for name in ("top_prompt", "center_space", "hook_bar", "press_sequence", "get_window"):
        left, top, right, bottom = normalized_to_pixel_roi(detector.roi_config.rois[name], width, height)
        crop = frame[top:bottom, left:right]
        crop = cv2.resize(crop, (260, 100), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        crops.append(crop)
    while len(crops) % 2:
        crops.append(np.zeros_like(crops[0]))
    rows = [cv2.hconcat(crops[index:index + 2]) for index in range(0, len(crops), 2)]
    cv2.imwrite(str(destination), cv2.vconcat(rows))


def _write_confusion(path: Path, matrix: dict[str, Counter[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["expected_state", *PREDICTED_STATES])
        for expected in EVALUATED_STATES:
            writer.writerow(
                [expected, *(matrix[expected].get(detected, 0) for detected in PREDICTED_STATES)]
            )


def _state_metrics(rows: list[dict[str, Any]], detected_key: str) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for state in EVALUATED_STATES:
        support = sum(row["expected_state"] == state for row in rows)
        predicted = sum(row[detected_key] == state for row in rows)
        true_positive = sum(
            row["expected_state"] == state and row[detected_key] == state for row in rows
        )
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        metrics[state] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def _format_metric(value: float | int | None) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _write_segments(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["start_frame", "end_frame", "state", "frame_count"])
        writer.writeheader()
        if not rows:
            return
        start = int(rows[0]["frame_index"])
        previous = start
        state = str(rows[0]["smoothed_state"] or rows[0]["detected_state"])
        for row in rows[1:]:
            frame_index = int(row["frame_index"])
            detected = str(row["smoothed_state"] or row["detected_state"])
            if detected != state:
                writer.writerow({"start_frame": start, "end_frame": previous, "state": state, "frame_count": previous - start + 1})
                start, state = frame_index, detected
            previous = frame_index
        writer.writerow({"start_frame": start, "end_frame": previous, "state": state, "frame_count": previous - start + 1})


def _write_report(
    session: ReplaySession,
    rows: list[dict[str, Any]],
    evaluated_rows: list[dict[str, Any]],
    raw_metrics: dict[str, dict[str, float | int | None]],
    smoothed_metrics: dict[str, dict[str, float | int | None]],
    hard_counts: Counter[str],
    annotations: dict[str, Any],
    hook_state_range: tuple[int, int] | None,
    hook_bar_range: tuple[int, int] | None,
    press_range: tuple[int, int] | None,
) -> Path:
    evaluated_count = len(evaluated_rows)
    ignored_count = len(rows) - evaluated_count
    raw_accuracy = (
        sum(bool(row["is_correct_raw"]) for row in evaluated_rows) / evaluated_count
        if evaluated_count
        else 0.0
    )
    smoothed_accuracy = (
        sum(bool(row["is_correct_smoothed"]) for row in evaluated_rows) / evaluated_count
        if evaluated_count
        else 0.0
    )
    unknown_ratio = (
        sum(row["detected_state"] == "UNKNOWN" for row in evaluated_rows) / evaluated_count
        if evaluated_count
        else 0.0
    )
    hook_rows = [row for row in evaluated_rows if row["expected_state"] == "HOOK"]
    hook_bar_rows = [
        row for row in rows if _in_range(int(row["frame_index"]), hook_bar_range)
    ]
    press_rows = [row for row in rows if _in_range(int(row["frame_index"]), press_range)]
    get_rows = [row for row in evaluated_rows if row["expected_state"] == "GET"]
    hook_state_label = (
        f"{hook_state_range[0]}-{hook_state_range[1]}" if hook_state_range else "N/A"
    )
    hook_bar_label = f"{hook_bar_range[0]}-{hook_bar_range[1]}" if hook_bar_range else "N/A"
    press_label = f"{press_range[0]}-{press_range[1]}" if press_range else "N/A"
    lines = [
        "# Replay Evaluation Report",
        "",
        f"- Session: `{session.path.name}`",
        f"- Total frames: {len(rows)}",
        f"- Evaluated frames: {evaluated_count}",
        f"- Ignored frames: {ignored_count}",
        f"- Overall accuracy: {smoothed_accuracy:.2%}",
        f"- Raw accuracy: {raw_accuracy:.2%}",
        f"- Smoothed accuracy: {smoothed_accuracy:.2%}",
        f"- UNKNOWN ratio: {unknown_ratio:.2%}",
        "",
        "## Per-state metrics",
        "",
        "| State | Support | Raw precision | Raw recall | Raw F1 | Smoothed precision | Smoothed recall | Smoothed F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for state in EVALUATED_STATES:
        raw = raw_metrics[state]
        smoothed = smoothed_metrics[state]
        lines.append(
            f"| {state} | {raw['support']} | {_format_metric(raw['precision'])} | "
            f"{_format_metric(raw['recall'])} | {_format_metric(raw['f1'])} | "
            f"{_format_metric(smoothed['precision'])} | {_format_metric(smoothed['recall'])} | "
            f"{_format_metric(smoothed['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic components",
            "",
            f"- Top-level HOOK state {hook_state_label}: {len(hook_rows)} expected; "
            f"raw HOOK {sum(row['detected_state'] == 'HOOK' for row in hook_rows)}; "
            f"smoothed HOOK {sum(row['smoothed_state'] == 'HOOK' for row in hook_rows)}",
            f"- HOOK bar-required frames {hook_bar_label}: {len(hook_bar_rows)}; "
            f"hook bar detected: {sum(bool(row['hook_detected']) for row in hook_bar_rows)}",
            f"- PRESS event {press_label}: {len(press_rows)}; press detected: "
            f"{sum(bool(row['press_detected']) for row in press_rows)}",
            f"- GET expected frames: {len(get_rows)}; get window detected: {sum(bool(row['get_detected']) for row in get_rows)}",
            f"- False GET: {sum(row['expected_state'] != 'GET' and row['detected_state'] == 'GET' for row in evaluated_rows)}",
            f"- False PRESS: {sum(row['expected_state'] != 'PRESS' and row['detected_state'] == 'PRESS' for row in evaluated_rows)}",
            f"- False HOOK: {sum(row['expected_state'] != 'HOOK' and row['detected_state'] == 'HOOK' for row in evaluated_rows)}",
            "",
            "## Hard examples exported",
            "",
        ]
    )
    lines.extend([f"- {category}: {count}" for category, count in sorted(hard_counts.items())] or ["- None"])
    lines.extend(["", "## Possible problems", ""])
    if unknown_ratio > 0.25:
        lines.append("- UNKNOWN ratio exceeds the repair target; compare hard examples and ROI crops before adjusting thresholds.")
    waiting_recall = smoothed_metrics["WAITING"]["recall"]
    if waiting_recall is not None and waiting_recall < 0.50:
        lines.append("- WAITING recall is low; top_prompt matching likely does not generalize from reference captures.")
    if annotations.get("notes", {}).get("get_skipped"):
        reason = annotations.get("notes", {}).get("reason", "GET was skipped in this session")
        lines.append(f"- GET skipped by annotation: {reason}")
    if not any(lines[-1].startswith("-") for _ in [0]):
        lines.append("- None")
    destination = session.path / "replay_eval_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def evaluate_session(session_path: str | Path, *, export_debug: bool = True) -> EvaluationSummary:
    session = ReplaySession.load(session_path)
    frames = session.frame_paths()
    labels = _load_ground_truth(session.path / "ground_truth.yaml", len(frames))
    annotations = load_annotations(session.path / "annotations.yaml", len(frames))
    hook_state_range = _event_range(
        annotations, "hook_state", _frame_range_for_state(labels, "HOOK")
    )
    hook_bar_range = _event_range(annotations, "hook_bar_visible", hook_state_range)
    press_range = _event_range(annotations, "press", _frame_range_for_state(labels, "PRESS"))
    detector = StateDetector(PROJECT_ROOT / "assets" / "reference")
    output_root = session.path / "debug"
    hard_dir, crops_dir, montage_dir = output_root / "hard_examples", output_root / "roi_crops", output_root / "montage"
    if export_debug:
        for directory in (hard_dir, crops_dir, montage_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    hard_counts: Counter[str] = Counter()
    hard_paths: list[Path] = []
    for frame_index, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Cannot read replay frame: {frame_path}")
        expected = labels[frame_index]
        is_evaluated = expected != IGNORE_STATE
        result = detector.detect_state(frame)
        hook = detect_hook_bar(frame, detector.roi_config, detector.thresholds, save_debug=False)
        press = detect_press_sequence(frame, detector.roi_config, detector.thresholds, save_debug=False)
        get = detect_get_window(frame, detector.roi_config, detector.thresholds)
        get_detected, get_confidence = get["detected"], get["confidence"]
        detected = result.state
        raw_correct: bool | str = detected == expected if is_evaluated else ""
        row = {
            "frame_index": frame_index,
            "filename": frame_path.name,
            "expected_state": expected,
            "detected_state": detected,
            "raw_detected_state": detected,
            "smoothed_state": "",
            "confidence": result.confidence,
            "is_evaluated": is_evaluated,
            "is_correct_raw": raw_correct,
            "is_correct_smoothed": "",
            "raw_scores": json.dumps(result.debug.get("raw_scores", {}), ensure_ascii=False, sort_keys=True),
            "hook_detected": hook["detected"],
            "hook_confidence": hook["confidence"],
            "hook_fill_ratio": hook["fill_ratio"] if hook["fill_ratio"] is not None else "",
            "hook_divider_ratio": hook["divider_ratio"] if hook["divider_ratio"] is not None else "",
            "press_detected": press["detected"],
            "press_confidence": press["confidence"],
            "press_sequence_text": press["sequence_text"],
            "get_detected": get_detected,
            "get_confidence": round(get_confidence, 4),
            "matched_features": "|".join(result.matched_features),
        }
        rows.append(row)
        if is_evaluated:
            matrix[expected][detected] += 1
        category = _hard_example_category(expected, detected) if is_evaluated else None
        if export_debug and category is not None and hard_counts[category] < 10:
            hard_counts[category] += 1
            annotated = _annotate_frame(frame, detector, row)
            hard_path = hard_dir / f"{category}_{frame_index:06d}.jpg"
            cv2.imwrite(str(hard_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            hard_paths.append(hard_path)
            crop_path = crops_dir / f"{category}_{frame_index:06d}_rois.jpg"
            _save_roi_montage(frame, detector, crop_path)

    if export_debug and hard_paths:
        thumbnails = []
        for hard_path in hard_paths[:36]:
            image = cv2.imread(str(hard_path), cv2.IMREAD_COLOR)
            if image is not None:
                thumbnails.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        if thumbnails:
            while len(thumbnails) % 3:
                thumbnails.append(np.zeros_like(thumbnails[0]))
            montage = cv2.vconcat([cv2.hconcat(thumbnails[index:index + 3]) for index in range(0, len(thumbnails), 3)])
            cv2.imwrite(str(montage_dir / "hard_examples_montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 85])

    smoothed_states = StateSmoother().smooth([str(row["detected_state"]) for row in rows], [float(row["confidence"]) for row in rows])
    for row, smoothed_state in zip(rows, smoothed_states, strict=True):
        row["smoothed_state"] = smoothed_state
        row["is_correct_smoothed"] = (
            smoothed_state == row["expected_state"] if row["is_evaluated"] else ""
        )

    fields = list(rows[0]) if rows else [
        "frame_index", "filename", "expected_state", "detected_state", "raw_detected_state", "smoothed_state", "confidence", "is_evaluated",
        "is_correct_raw", "is_correct_smoothed", "raw_scores", "hook_detected", "hook_confidence", "hook_fill_ratio",
        "hook_divider_ratio", "press_detected", "press_confidence", "press_sequence_text", "get_detected", "get_confidence", "matched_features",
    ]
    results_path = session.path / "replay_eval_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_confusion(session.path / "confusion_matrix.csv", matrix)
    _write_segments(session.path / "state_segments_predicted.csv", rows)
    evaluated_rows = [row for row in rows if row["is_evaluated"]]
    raw_metrics = _state_metrics(evaluated_rows, "detected_state")
    smoothed_metrics = _state_metrics(evaluated_rows, "smoothed_state")
    report_path = _write_report(
        session,
        rows,
        evaluated_rows,
        raw_metrics,
        smoothed_metrics,
        hard_counts,
        annotations,
        hook_state_range,
        hook_bar_range,
        press_range,
    )
    evaluated_count = len(evaluated_rows)
    raw_accuracy = (
        sum(bool(row["is_correct_raw"]) for row in evaluated_rows) / evaluated_count
        if evaluated_count
        else 0.0
    )
    smoothed_accuracy = (
        sum(bool(row["is_correct_smoothed"]) for row in evaluated_rows) / evaluated_count
        if evaluated_count
        else 0.0
    )
    unknown_ratio = (
        sum(row["detected_state"] == "UNKNOWN" for row in evaluated_rows) / evaluated_count
        if evaluated_count
        else 0.0
    )
    state_recalls = {
        state: (
            float(smoothed_metrics[state]["recall"])
            if smoothed_metrics[state]["recall"] is not None
            else None
        )
        for state in EVALUATED_STATES
    }
    return EvaluationSummary(
        session_path=session.path,
        results_path=results_path,
        report_path=report_path,
        total_frames=len(rows),
        evaluated_frames=evaluated_count,
        ignored_frames=len(rows) - evaluated_count,
        raw_accuracy=raw_accuracy,
        smoothed_accuracy=smoothed_accuracy,
        unknown_ratio=unknown_ratio,
        state_recalls=state_recalls,
        hook_detected_count=sum(
            bool(row["hook_detected"])
            for row in rows
            if _in_range(int(row["frame_index"]), hook_bar_range)
        ),
        press_detected_count=sum(
            bool(row["press_detected"])
            for row in rows
            if _in_range(int(row["frame_index"]), press_range)
        ),
        get_detected_count=sum(
            bool(row["get_detected"])
            for row in evaluated_rows
            if row["expected_state"] == "GET"
        ),
        hook_bar_range=hook_bar_range,
        press_range=press_range,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing replay session against ground_truth.yaml.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", type=Path)
    selection.add_argument("--latest", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--no-debug-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_path = args.session if args.session else latest_session(args.session_root)
    summary = evaluate_session(session_path, export_debug=not args.no_debug_export)
    print(f"session: {summary.session_path}")
    print(f"total_frames: {summary.total_frames}")
    print(f"evaluated_frames: {summary.evaluated_frames}")
    print(f"ignored_frames: {summary.ignored_frames}")
    print(f"raw_accuracy: {summary.raw_accuracy:.2%}")
    print(f"smoothed_accuracy: {summary.smoothed_accuracy:.2%}")
    print(f"unknown_ratio: {summary.unknown_ratio:.2%}")
    print(f"state_recalls: {summary.state_recalls}")
    hook_range = (
        f"{summary.hook_bar_range[0]}_{summary.hook_bar_range[1]}"
        if summary.hook_bar_range
        else "not_annotated"
    )
    press_range = (
        f"{summary.press_range[0]}_{summary.press_range[1]}"
        if summary.press_range
        else "not_annotated"
    )
    print(f"hook_detected_{hook_range}: {summary.hook_detected_count}")
    print(f"press_detected_{press_range}: {summary.press_detected_count}")
    print(f"get_detected_expected_frames: {summary.get_detected_count}")
    print(f"results: {summary.results_path}")
    print(f"report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

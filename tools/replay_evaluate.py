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
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import ROI_NAMES, normalized_to_pixel_roi  # noqa: E402
from src.detectors.hook_detector import detect_hook_bar  # noqa: E402
from src.detectors.press_detector import detect_press_sequence  # noqa: E402
from src.detectors.get_detector import detect_get_window  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402
from src.state_smoother import StateSmoother  # noqa: E402
from src.state_detector import ROI_NAMES_BY_STATE, StateDetector  # noqa: E402


STATES = (*ROI_NAMES_BY_STATE, "UNKNOWN")


@dataclass(frozen=True)
class EvaluationSummary:
    session_path: Path
    results_path: Path
    report_path: Path
    raw_accuracy: float
    smoothed_accuracy: float
    unknown_ratio: float
    state_recalls: dict[str, float]
    hook_detected_count: int
    press_detected_count: int
    get_detected_count: int


def _load_ground_truth(path: Path, frame_count: int) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {path}; run tools/create_ground_truth.py first")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError(f"Ground truth must contain a segments list: {path}")
    labels: dict[int, str] = {}
    for segment in data["segments"]:
        if not isinstance(segment, dict):
            raise ValueError("Every ground-truth segment must be a mapping")
        start, end, state = segment.get("start"), segment.get("end"), segment.get("state")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(state, str):
            raise ValueError("Ground-truth segments require integer start/end and string state")
        state = state.upper()
        if state not in ROI_NAMES_BY_STATE or start < 1 or end < start or end > frame_count:
            raise ValueError(f"Invalid ground-truth segment: {segment}")
        for frame_index in range(start, end + 1):
            if frame_index in labels:
                raise ValueError(f"Overlapping ground-truth label at frame {frame_index}")
            labels[frame_index] = state
    return labels


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
        writer.writerow(["expected_state", *STATES])
        for expected in STATES:
            writer.writerow([expected, *(matrix[expected].get(detected, 0) for detected in STATES)])


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
    matrix: dict[str, Counter[str]],
    raw_recalls: dict[str, float],
    smoothed_recalls: dict[str, float],
    hard_counts: Counter[str],
) -> Path:
    raw_accuracy = sum(bool(row["is_correct_raw"]) for row in rows) / len(rows) if rows else 0.0
    smoothed_accuracy = sum(bool(row["is_correct_smoothed"]) for row in rows) / len(rows) if rows else 0.0
    unknown_ratio = sum(row["detected_state"] == "UNKNOWN" for row in rows) / len(rows) if rows else 0.0
    hook_rows = [row for row in rows if row["expected_state"] == "HOOK"]
    press_rows = [row for row in rows if row["expected_state"] == "PRESS"]
    get_rows = [row for row in rows if row["expected_state"] == "GET"]
    lines = [
        "# Replay Evaluation Report",
        "",
        f"- Session: `{session.path.name}`",
        f"- Evaluated frames: {len(rows)}",
        f"- Raw accuracy: {raw_accuracy:.2%}",
        f"- Smoothed accuracy: {smoothed_accuracy:.2%}",
        f"- UNKNOWN ratio: {unknown_ratio:.2%}",
        "",
        "## Per-state recall (raw / smoothed)",
        "",
    ]
    lines.extend([f"- {state}: {raw_recalls.get(state, 0.0):.2%} / {smoothed_recalls.get(state, 0.0):.2%}" for state in ROI_NAMES_BY_STATE])
    lines.extend(
        [
            "",
            "## Diagnostic components",
            "",
            f"- HOOK expected frames: {len(hook_rows)}; hook bar detected: {sum(bool(row['hook_detected']) for row in hook_rows)}",
            f"- HOOK bar-required frames 459-471: {sum(bool(row['hook_detected']) for row in hook_rows if 459 <= int(row['frame_index']) <= 471)}",
            f"- PRESS expected frames: {len(press_rows)}; press detected: {sum(bool(row['press_detected']) for row in press_rows)}",
            f"- GET expected frames: {len(get_rows)}; get window detected: {sum(bool(row['get_detected']) for row in get_rows)}",
            f"- False GET: {sum(row['expected_state'] != 'GET' and row['detected_state'] == 'GET' for row in rows)}",
            f"- False PRESS: {sum(row['expected_state'] != 'PRESS' and row['detected_state'] == 'PRESS' for row in rows)}",
            f"- False HOOK: {sum(row['expected_state'] != 'HOOK' and row['detected_state'] == 'HOOK' for row in rows)}",
            "",
            "## Hard examples exported",
            "",
        ]
    )
    lines.extend([f"- {category}: {count}" for category, count in sorted(hard_counts.items())] or ["- None"])
    lines.extend(["", "## Possible problems", ""])
    if unknown_ratio > 0.25:
        lines.append("- UNKNOWN ratio exceeds the repair target; compare hard examples and ROI crops before adjusting thresholds.")
    if smoothed_recalls.get("WAITING", 0.0) < 0.50:
        lines.append("- WAITING recall is low; top_prompt matching likely does not generalize from reference captures.")
    if not any(lines[-1].startswith("-") for _ in [0]):
        lines.append("- None")
    destination = session.path / "replay_eval_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def evaluate_session(session_path: str | Path, *, export_debug: bool = True) -> EvaluationSummary:
    session = ReplaySession.load(session_path)
    frames = session.frame_paths()
    labels = _load_ground_truth(session.path / "ground_truth.yaml", len(frames))
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
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    hard_counts: Counter[str] = Counter()
    hard_paths: list[Path] = []
    for frame_index, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Cannot read replay frame: {frame_path}")
        expected = labels.get(frame_index, "UNLABELED")
        result = detector.detect_state(frame)
        hook = detect_hook_bar(frame, detector.roi_config, detector.thresholds, save_debug=False)
        press = detect_press_sequence(frame, detector.roi_config, detector.thresholds, save_debug=False)
        get = detect_get_window(frame, detector.roi_config, detector.thresholds)
        get_detected, get_confidence = get["detected"], get["confidence"]
        detected = result.state
        raw_correct = expected != "UNLABELED" and detected == expected
        row = {
            "frame_index": frame_index,
            "filename": frame_path.name,
            "expected_state": expected,
            "detected_state": detected,
            "raw_detected_state": detected,
            "smoothed_state": "",
            "confidence": result.confidence,
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
        if expected != "UNLABELED":
            totals[expected] += 1
            correct[expected] += int(raw_correct)
            matrix[expected][detected] += 1
        category = _hard_example_category(expected, detected) if expected != "UNLABELED" else None
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
        row["is_correct_smoothed"] = row["expected_state"] != "UNLABELED" and smoothed_state == row["expected_state"]

    fields = list(rows[0]) if rows else [
        "frame_index", "filename", "expected_state", "detected_state", "raw_detected_state", "smoothed_state", "confidence",
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
    raw_recalls = {state: correct[state] / totals[state] if totals[state] else 0.0 for state in ROI_NAMES_BY_STATE}
    smoothed_correct: Counter[str] = Counter()
    for row in rows:
        if row["expected_state"] != "UNLABELED":
            smoothed_correct[str(row["expected_state"])] += int(bool(row["is_correct_smoothed"]))
    smoothed_recalls = {state: smoothed_correct[state] / totals[state] if totals[state] else 0.0 for state in ROI_NAMES_BY_STATE}
    report_path = _write_report(session, rows, matrix, raw_recalls, smoothed_recalls, hard_counts)
    raw_accuracy = sum(bool(row["is_correct_raw"]) for row in rows) / len(rows) if rows else 0.0
    smoothed_accuracy = sum(bool(row["is_correct_smoothed"]) for row in rows) / len(rows) if rows else 0.0
    unknown_ratio = sum(row["detected_state"] == "UNKNOWN" for row in rows) / len(rows) if rows else 0.0
    return EvaluationSummary(
        session.path,
        results_path,
        report_path,
        raw_accuracy,
        smoothed_accuracy,
        unknown_ratio,
        smoothed_recalls,
        sum(bool(row["hook_detected"]) for row in rows if 459 <= int(row["frame_index"]) <= 471),
        sum(bool(row["press_detected"]) for row in rows if 472 <= int(row["frame_index"]) <= 483),
        sum(bool(row["get_detected"]) for row in rows if 484 <= int(row["frame_index"]) <= 499),
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
    print(f"raw_accuracy: {summary.raw_accuracy:.2%}")
    print(f"smoothed_accuracy: {summary.smoothed_accuracy:.2%}")
    print(f"unknown_ratio: {summary.unknown_ratio:.2%}")
    print(f"state_recalls: {summary.state_recalls}")
    print(f"hook_detected_459_471: {summary.hook_detected_count}")
    print(f"press_detected_472_483: {summary.press_detected_count}")
    print(f"get_detected_484_499: {summary.get_detected_count}")
    print(f"results: {summary.results_path}")
    print(f"report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate read-only Hook/PRESS/GET diagnostics for one Pilot replay session."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_roi_config, normalized_to_pixel_roi  # noqa: E402
from src.detectors.get_detector import detect_get_window  # noqa: E402
from src.detectors.hook_detector import detect_hook_bar  # noqa: E402
from src.detectors.press_detector import detect_press_sequence  # noqa: E402
from src.fishing_v2.domain.observations import HookObservation  # noqa: E402
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode  # noqa: E402
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier  # noqa: E402


DEFAULT_SESSION = "session_20260710_130308"
DEFAULT_SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
DEFAULT_OUTPUT = ROOT / "reports" / "fishing_v2" / "pilot_detector_diagnostics"
SUMMARY_MD = ROOT / "reports" / "fishing_v2" / "pilot_detector_diagnostics_summary.md"
SUMMARY_JSON = ROOT / "reports" / "fishing_v2" / "pilot_detector_diagnostics_summary.json"


def _labels(path: Path, frame_count: int) -> dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for segment in data["segments"]:
        for frame in range(int(segment["start"]), int(segment["end"]) + 1):
            labels[frame] = str(segment["state"])
    if set(labels) != set(range(1, frame_count + 1)):
        raise ValueError("Pilot global ground truth coverage is incomplete")
    return labels


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def _tile(crop: np.ndarray, lines: Sequence[str], *, width: int = 480, image_height: int = 150) -> np.ndarray:
    content = _fit(crop, width, image_height)
    margin = 22 * min(5, len(lines)) + 8
    output = np.zeros((image_height + margin, width, 3), dtype=np.uint8)
    output[:image_height] = content
    y = image_height + 18
    for line in lines[:5]:
        cv2.putText(output, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
        y += 21
    return output


def _sheet(tiles: Sequence[np.ndarray], columns: int = 3) -> np.ndarray:
    if not tiles:
        return np.zeros((200, 480, 3), dtype=np.uint8)
    height, width = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    output = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        output[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    return output


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"Could not write {path}")


def _sample(rows: list[dict[str, Any]], maximum: int = 6) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return rows
    indexes = np.linspace(0, len(rows) - 1, maximum)
    return [rows[round(float(index))] for index in indexes]


def _press_rejections(result: dict[str, Any]) -> list[str]:
    debug = result["debug"]
    reasons: list[str] = []
    if not debug["template_keys"]:
        reasons.append("templates_unavailable")
    if not result["panel_present"]:
        reasons.append(result["panel_qualification_reason"])
    if not result["key_boxes"]:
        reasons.append("sequence_candidate_missing")
    if any(key not in "WASD" for key in result["sequence_candidate"]):
        reasons.append("invalid_key_classification")
    if result["sequence_confidence"] < 0.68:
        reasons.append(f"sequence_confidence:{result['sequence_confidence']:.4f}<0.68")
    if result["panel_present"] and not result["sequence_ready"]:
        reasons.append(
            "panel_confirmation_required_for_arrow_freeze"
            if result["debug"].get("arrow_sequence_ready")
            else "temporal_sequence_consensus_required"
        )
    return reasons or ["panel_and_sequence_ready"]


def _get_rejections(result: dict[str, Any]) -> list[str]:
    scores = result["debug"].get("feature_scores", {})
    thresholds = {"inventory_title": 0.80, "item_grid": 0.70, "collect_button": 0.78}
    reasons = [
        f"{name}:{scores.get(name, 0.0):.4f}<{threshold:.2f}"
        for name, threshold in thresholds.items()
        if scores.get(name, 0.0) < threshold
    ]
    passed = [name for name in thresholds if name not in {item.split(":", 1)[0] for item in reasons}]
    if len(passed) < 2:
        reasons.append("fewer_than_two_feature_regions")
    if not ({"item_grid", "collect_button"} & set(passed)):
        reasons.append("no_grid_or_collect_button_confirmation")
    return reasons or ["accepted"]


def run(session_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest["frame_count"])
    extension = str(manifest["image_format"])
    labels = _labels(session_path / "ground_truth.yaml", frame_count)
    roi = load_roi_config()
    qualifier = DetectorEvidenceQualifier()
    hook_rows: list[dict[str, Any]] = []
    hook_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    press_rows: list[dict[str, Any]] = []
    get_rows: list[dict[str, Any]] = []

    for frame_index in range(1, frame_count + 1):
        frame_path = session_path / "frames" / f"{frame_index:06d}.{extension}"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        global_state = labels[frame_index]
        raw_hook = detect_hook_bar(frame, save_debug=False)
        hook_observation = HookObservation(
            bool(raw_hook["detected"]),
            float(raw_hook["confidence"]),
            frame_index,
            frame_index * float(manifest["interval_sec"]),
            fill_ratio=raw_hook.get("fill_ratio"),
            divider_ratio=raw_hook.get("divider_ratio"),
            evidence={"matched_features": list(raw_hook.get("matched_features", []))},
        )
        _, hook_qualification = qualifier.qualify_hook(
            hook_observation,
            DetectorActivationMode.BURST,
        )
        hook_row = {
            "frame": frame_index,
            "global_state": global_state,
            "raw_confidence": raw_hook["confidence"],
            "fill_ratio": raw_hook.get("fill_ratio"),
            "matched_features": raw_hook.get("matched_features", []),
            "raw_detected": raw_hook["detected"],
            "qualified_detected": hook_qualification.qualified_detected,
            "qualification_reason": hook_qualification.qualification_reason,
            "hook_evidence_kind": hook_qualification.hook_evidence_kind.value,
            "frame_path": str(frame_path),
        }
        hook_rows.append(hook_row)
        if raw_hook["detected"]:
            if global_state in {"READY", "WAITING", "IDLE"} and hook_qualification.hook_evidence_kind.value == "RECTANGLE_CANDIDATE":
                hook_groups[f"{global_state.lower()}_rectangle_false_positive"].append(hook_row)
            if global_state == "HOOK":
                if hook_qualification.hook_evidence_kind.value == "RECTANGLE_CANDIDATE":
                    hook_groups["hook_rectangle_only_stage"].append(hook_row)
                if "bar_fill" in raw_hook.get("matched_features", []):
                    hook_groups["hook_bar_fill_stage"].append(hook_row)
                if "divider_line" in raw_hook.get("matched_features", []):
                    hook_groups["hook_divider_stage"].append(hook_row)

        if 415 <= frame_index <= 438:
            raw_press = detect_press_sequence(frame, save_debug=False)
            press_rows.append({
                "frame": frame_index,
                "confidence": raw_press["confidence"],
                "detected": raw_press["detected"],
                "panel_candidate": raw_press["panel_candidate"],
                "panel_present": raw_press["panel_present"],
                "panel_qualification_reason": raw_press["panel_qualification_reason"],
                "sequence": raw_press["sequence_candidate"],
                "sequence_ready": raw_press["sequence_ready"],
                "clean_frame_eligible": raw_press["clean_frame_eligible"],
                "arrow_sequence_ready": raw_press["debug"].get("arrow_sequence_ready", False),
                "sequence_confidence": raw_press["sequence_confidence"],
                "key_boxes": raw_press["key_boxes"],
                "matched_features": raw_press["matched_features"],
                "dark_panel_ratio": raw_press["debug"]["dark_panel_ratio"],
                "colour_mode": raw_press["debug"]["colour_mode"],
                "detector_threshold": raw_press["debug"]["detector_min_confidence"],
                "rejection_reasons": _press_rejections(raw_press),
                "frame_path": str(frame_path),
            })

        if 435 <= frame_index <= 463:
            raw_get = detect_get_window(frame)
            get_rows.append({
                "frame": frame_index,
                "global_state": global_state,
                "detected": raw_get["detected"],
                "confidence": raw_get["confidence"],
                "feature_scores": raw_get["debug"].get("feature_scores", {}),
                "matched_features": raw_get["matched_features"],
                "detector_threshold": raw_get["debug"].get("detector_min_confidence"),
                "rejection_reasons": _get_rejections(raw_get),
                "frame_path": str(frame_path),
            })

    hook_bounds = normalized_to_pixel_roi(roi.rois["hook_bar_precise"], 2560, 1440)
    press_bounds = normalized_to_pixel_roi(roi.rois["press_sequence"], 2560, 1440)
    get_bounds = normalized_to_pixel_roi(roi.rois["get_window"], 2560, 1440)
    for group, rows in hook_groups.items():
        tiles = []
        for item in _sample(rows):
            frame = cv2.imread(item["frame_path"], cv2.IMREAD_COLOR)
            x1, y1, x2, y2 = hook_bounds
            tiles.append(_tile(frame[y1:y2, x1:x2], (
                f"frame={item['frame']} global={item['global_state']} raw_conf={item['raw_confidence']}",
                f"fill={item['fill_ratio']} features={','.join(item['matched_features'])}",
                f"raw={item['raw_detected']} qualified={item['qualified_detected']}",
                f"reason={item['qualification_reason']}",
            )))
        _write_image(output / "hook" / f"{group}.jpg", _sheet(tiles))

    press_tiles = []
    for item in press_rows:
        frame = cv2.imread(item["frame_path"], cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = press_bounds
        press_tiles.append(_tile(frame[y1:y2, x1:x2], (
            f"frame={item['frame']} confidence={item['confidence']} detected={item['detected']}",
            f"sequence={''.join(item['sequence']) or '-'} boxes={len(item['key_boxes'])}",
            f"features={','.join(item['matched_features']) or '-'} dark={item['dark_panel_ratio']}",
            f"mode={item['colour_mode']} threshold={item['detector_threshold']}",
            f"reject={';'.join(item['rejection_reasons'])}",
        )))
    _write_image(output / "press" / "frames_415_438.jpg", _sheet(press_tiles))

    get_tiles = []
    for item in get_rows:
        frame = cv2.imread(item["frame_path"], cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = get_bounds
        scores = item["feature_scores"]
        get_tiles.append(_tile(frame[y1:y2, x1:x2], (
            f"frame={item['frame']} global={item['global_state']} detected={item['detected']}",
            f"confidence={item['confidence']} threshold={item['detector_threshold']}",
            f"title={scores.get('inventory_title')} grid={scores.get('item_grid')}",
            f"button={scores.get('collect_button')} features={','.join(item['matched_features']) or '-'}",
            f"reject={';'.join(item['rejection_reasons'])}",
        ), image_height=210))
    _write_image(output / "get" / "frames_435_463.jpg", _sheet(get_tiles))

    output.mkdir(parents=True, exist_ok=True)
    (output / "hook" / "details.json").write_text(json.dumps(hook_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "press" / "details.json").write_text(json.dumps(press_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "get" / "details.json").write_text(json.dumps(get_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_by_state = Counter(item["global_state"] for item in hook_rows if item["raw_detected"])
    qualified_by_state = Counter(item["global_state"] for item in hook_rows if item["qualified_detected"])
    non_hook_qualified = sum(count for state, count in qualified_by_state.items() if state != "HOOK")
    press_428 = next(item for item in press_rows if item["frame"] == 428)
    get_true = [item for item in get_rows if 440 <= item["frame"] <= 458]
    press_reason_counts = Counter(reason.split(":", 1)[0] for item in press_rows for reason in item["rejection_reasons"])
    get_reason_counts = Counter(reason.split(":", 1)[0] for item in get_true for reason in item["rejection_reasons"])
    summary = {
        "session": session_path.name,
        "hook": {
            "raw_detected_by_global_state": dict(raw_by_state),
            "qualified_active_by_global_state": dict(qualified_by_state),
            "non_hook_qualified_false_positive_count": non_hook_qualified,
            "qualification": "bar_fill + positive fill_ratio + strong confidence; divider optional",
        },
        "press": {
            "range": [415, 438],
            "panel_present": sum(item["panel_present"] for item in press_rows),
            "sequence_ready": sum(item["sequence_ready"] for item in press_rows),
            "clean_arrow_candidate_frames": sum(item["arrow_sequence_ready"] for item in press_rows),
            "total": len(press_rows),
            "rejection_reason_counts": dict(press_reason_counts),
            "frame_428": {key: value for key, value in press_428.items() if key != "frame_path"},
            "temporal_condition_used": True,
        },
        "get": {
            "visual_panel_range": [440, 458],
            "diagnostic_range": [435, 463],
            "original_detected_in_visual_range": 0,
            "original_rejection_reason": (
                "fixed whole-ROI template comparison was misaligned with the live panel scale and position"
            ),
            "detected_in_visual_range": sum(item["detected"] for item in get_true),
            "total_visual_frames": len(get_true),
            "rejection_reason_counts": dict(get_reason_counts),
            "visual_disappearance_frame": 459,
        },
        "thresholds_modified": False,
        "output_root": str(output),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Pilot Detector Diagnostics Summary",
        "",
        f"- Session: `{session_path.name}`",
        "- Detector thresholds modified: **false**",
        f"- Hook raw detected by global state: `{dict(raw_by_state)}`",
        f"- Hook qualified ACTIVE by global state: `{dict(qualified_by_state)}`",
        f"- Non-HOOK qualified false positives: **{non_hook_qualified}**",
        "- Hook qualification: raw detected + `bar_fill` + positive `fill_ratio` + strong confidence; divider is optional.",
        f"- PRESS panel present: **{summary['press']['panel_present']}/{summary['press']['total']}**",
        f"- PRESS sequence ready (single-frame tool): **{summary['press']['sequence_ready']}/{summary['press']['total']}**",
        f"- PRESS clean arrow candidate frames: **{summary['press']['clean_arrow_candidate_frames']}/{summary['press']['total']}**",
        f"- PRESS rejection counts: `{dict(press_reason_counts)}`",
        f"- Frame 428: panel_confidence={press_428['confidence']}, sequence_candidate={press_428['sequence']}, sequence_confidence={press_428['sequence_confidence']}, boxes={len(press_428['key_boxes'])}, reasons=`{press_428['rejection_reasons']}`.",
        "- Runtime sequence readiness is temporal; see `press_detector_diagnostics_summary.md` and the manual review bundle.",
        "- GET original baseline: **0/19**; the fixed whole-ROI template comparison was misaligned with the live panel scale and position.",
        f"- GET detected in visual panel range 440-458: **{summary['get']['detected_in_visual_range']}/{len(get_true)}**",
        f"- GET rejection counts: `{dict(get_reason_counts)}`",
        "- Visual boundary supplied by manual review: panel appears at frame 440 and disappears at frame 459.",
        "",
        "Large diagnostic sheets/details are under `reports/fishing_v2/pilot_detector_diagnostics/` and are Git-ignored.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pilot Hook/PRESS/GET detector diagnostics without changing thresholds.")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = Path(args.session)
    session_path = session if session.is_dir() else args.session_root / args.session
    summary = run(session_path, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

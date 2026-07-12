"""Generate a small GET-only diagnostic bundle for reviewed replay intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_roi_config, normalized_to_pixel_roi  # noqa: E402
from src.detectors.get_detector import detect_get_window  # noqa: E402
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner  # noqa: E402
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode  # noqa: E402


CASES = {
    "success_130308": {
        "session": "session_20260710_130308", "start": 440, "end": 459,
        "images": (440, 458, 459),
    },
    "false_negative_192315": {
        "session": "session_20260709_192315", "start": 484, "end": 499,
        "images": (484, 485, 499),
    },
    "false_positive_124419_316": {
        "session": "session_20260710_124419", "start": 306, "end": 326,
        "images": (306, 316, 326),
    },
    "false_positive_125441_9": {
        "session": "session_20260710_125441", "start": 1, "end": 19,
        "images": (1, 9, 19),
    },
    "false_positive_125441_394": {
        "session": "session_20260710_125441", "start": 384, "end": 404,
        "images": (384, 394, 404),
    },
    "false_positive_131254_1": {
        "session": "session_20260710_131254", "start": 1, "end": 35,
        "images": (1, 24, 35),
    },
}


def _write_images(
    frame: Any,
    detector: dict[str, Any],
    destination: Path,
) -> None:
    roi = load_roi_config().rois["get_window"]
    left, top, right, bottom = normalized_to_pixel_roi(
        roi, frame.shape[1], frame.shape[0]
    )
    debug = detector["debug"]
    bbox = debug.get("panel_bbox")
    fallback = (debug.get("fixed_fallback") or {}).get("bbox")
    marked = frame.copy()
    cv2.rectangle(marked, (left, top), (right, bottom), (255, 180, 0), 4)
    active = bbox or fallback
    if active:
        x1, y1, x2, y2 = (int(value) for value in active)
        colour = (0, 220, 0) if bbox else (0, 140, 255)
        cv2.rectangle(marked, (left + x1, top + y1), (left + x2, top + y2), colour, 4)
    cv2.putText(
        marked,
        f"GET={detector['detected']} conf={detector['confidence']:.3f} "
        f"source={debug.get('localization_source')}",
        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
    )
    scale = 800 / marked.shape[1]
    thumb = cv2.resize(marked, (800, round(marked.shape[0] * scale)))
    cv2.imwrite(str(destination.with_name(destination.name + "_full.jpg")), thumb)
    roi_crop = frame[top:bottom, left:right].copy()
    if active:
        x1, y1, x2, y2 = (int(value) for value in active)
        colour = (0, 220, 0) if bbox else (0, 140, 255)
        cv2.rectangle(roi_crop, (x1, y1), (x2, y2), colour, 2)
    cv2.imwrite(str(destination.with_name(destination.name + "_roi.jpg")), roi_crop)
    if active:
        x1, y1, x2, y2 = (int(value) for value in active)
        panel = frame[top + y1:top + y2, left + x1:left + x2]
        if panel.size:
            cv2.imwrite(str(destination.with_name(destination.name + "_panel.jpg")), panel)


def _compact_frame(
    frame_index: int,
    detector: dict[str, Any],
    replay_row: dict[str, Any],
) -> dict[str, Any]:
    debug = detector["debug"]
    expected_get = replay_row["global_ground_truth"] == "GET"
    raw = bool(detector["detected"])
    outcome = (
        "true_positive" if expected_get and raw
        else "false_negative" if expected_get
        else "false_positive" if raw
        else "true_negative"
    )
    return {
        "frame": frame_index,
        "global_state_offline_only": replay_row["global_ground_truth"],
        "runtime_state": replay_row["next_runtime_state"],
        "outcome": outcome,
        "panel_bbox": debug.get("panel_bbox"),
        "localization_source": debug.get("localization_source"),
        "feature_scores": debug.get("feature_scores"),
        "similarity_scores": debug.get("similarity_scores"),
        "structure_scores": debug.get("structure_scores"),
        "structure_debug": debug.get("structure_debug"),
        "total_confidence": detector["confidence"],
        "raw_detected": raw,
        "qualified_detected": bool(replay_row["qualified_detected"]["get"]),
        "used_by_fusion": bool(replay_row["used_by_fusion"]["get"]),
        "qualification_reason": replay_row["qualification_reason"]["get"],
    }


def generate(
    *, session_root: Path, config: Path, output_root: Path, work_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    runner = V2ReplayRunner(config)
    runs: dict[str, Any] = {}
    for session_id in sorted({str(case["session"]) for case in CASES.values()}):
        runs[session_id] = runner.run(
            session_root / session_id,
            mode="scripted_prompt",
            report_dir=work_root,
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            flat_report=False,
        )
    report: dict[str, Any] = {"cases": {}}
    for case_name, case in CASES.items():
        session_id = str(case["session"])
        run = runs[session_id]
        rows = {int(row["frame_index"]): row for row in run.rows}
        frames: list[dict[str, Any]] = []
        for frame_index in range(int(case["start"]), int(case["end"]) + 1):
            path = session_root / session_id / "frames" / f"{frame_index:06d}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            detector = detect_get_window(image)
            frames.append(_compact_frame(frame_index, detector, rows[frame_index]))
            if frame_index in case["images"]:
                _write_images(image, detector, output_root / f"{case_name}_{frame_index:06d}")
        report["cases"][case_name] = {
            "session": session_id,
            "range": [case["start"], case["end"]],
            "frames": frames,
        }
        print(f"{case_name}: {session_id} {case['start']}-{case['end']}", flush=True)
    destination = output_root / "diagnostics.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate focused GET replay diagnostics.")
    parser.add_argument(
        "--session-root", type=Path,
        default=PROJECT_ROOT / "assets" / "replay" / "sessions",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "reports" / "fishing_v2" / "get_cross_session_diagnostics",
    )
    parser.add_argument(
        "--work-root", type=Path,
        default=PROJECT_ROOT / "tmp" / "get_cross_session_diagnostics_replay",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate(
        session_root=args.session_root,
        config=args.config,
        output_root=args.output_root,
        work_root=args.work_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

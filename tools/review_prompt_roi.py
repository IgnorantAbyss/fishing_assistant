"""Generate manually reviewed Prompt ROI candidate contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_annotation import load_prompt_ground_truth  # noqa: E402
from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_roi_candidates  # noqa: E402
from src.ml.failure_analysis import uniform_sample  # noqa: E402


DEFAULT_SESSION_ROOT = PROJECT_ROOT / "assets" / "replay" / "sessions"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "fishing_v2" / "roi_review"
GLOBAL_STATES = ("IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET")


def _global_labels(path: Path, frame_count: int) -> dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for segment in data["segments"]:
        for frame in range(int(segment["start"]), int(segment["end"]) + 1):
            labels[frame] = str(segment["state"])
    if len(labels) != frame_count:
        raise ValueError(f"Incomplete global ground truth: {path}")
    return labels


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def review_prompt_roi(
    session_paths: Sequence[Path],
    candidates: Sequence[PromptROICandidate],
    report_dir: Path,
    *,
    samples_per_state: int = 4,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        tiles: list[np.ndarray] = []
        pixel_sizes: set[tuple[int, int]] = set()
        states_seen: set[str] = set()
        for session_path in session_paths:
            manifest = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
            count = int(manifest["frame_count"])
            extension = str(manifest["image_format"])
            global_labels = _global_labels(session_path / "ground_truth.yaml", count)
            prompt_path = session_path / "prompt_ground_truth.yaml"
            prompt_labels = load_prompt_ground_truth(prompt_path, count) if prompt_path.is_file() else {}
            by_state: dict[str, list[int]] = defaultdict(list)
            for frame, state in global_labels.items():
                if state in GLOBAL_STATES:
                    by_state[state].append(frame)
            selected = [frame for state in GLOBAL_STATES for frame in uniform_sample(by_state[state], samples_per_state)]
            for frame_index in selected:
                frame_path = session_path / "frames" / f"{frame_index:06d}.{extension}"
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise FileNotFoundError(frame_path)
                left, top, right, bottom = candidate.pixel_bounds(frame.shape[1], frame.shape[0])
                pixel_sizes.add((right - left, bottom - top))
                annotated = frame.copy()
                cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 4)
                crop = frame[top:bottom, left:right]
                tile = np.zeros((210, 800, 3), dtype=np.uint8)
                tile[:150, :390] = _fit(annotated, 390, 150)
                tile[:150, 410:] = _fit(crop, 390, 150)
                state = global_labels[frame_index]
                states_seen.add(state)
                prompt = prompt_labels.get(frame_index)
                cv2.putText(tile, f"{session_path.name} #{frame_index} global={state}", (4, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(tile, f"prompt={prompt.value if prompt else 'not_annotated'} candidate={candidate.candidate_id}", (4, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1)
                tiles.append(tile)
        sheet = np.vstack(tiles) if tiles else np.zeros((180, 800, 3), dtype=np.uint8)
        destination = report_dir / f"{candidate.candidate_id}.jpg"
        cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
        candidate_reports.append({
            "candidate_id": candidate.candidate_id,
            "normalized_coordinates": list(candidate.normalized),
            "pixel_sizes": [list(item) for item in sorted(pixel_sizes)],
            "states_covered": sorted(states_seen),
            "contains_complete_text": "manual_review_required",
            "contains_bottom_ui": "manual_review_required",
            "contains_large_scene_background": "manual_review_required",
            "covers_different_prompt_lengths": "manual_review_required",
            "manual_review_required": True,
            "contact_sheet": str(destination),
            "note": candidate.note,
        })
    return {
        "roi_status": "unapproved",
        "automatic_selection_performed": False,
        "bright_mask_used_for_approval": False,
        "candidates": candidate_reports,
    }


def _write_report(report: dict[str, Any], destination: Path) -> None:
    lines = [
        "# Prompt ROI Candidate Review",
        "",
        f"- ROI status: **{report['roi_status']}**",
        "- Automatic selection: false",
        "- Bright-mask approval: false",
        "- Final approval must be entered explicitly by the user in config/fishing_v2.yaml.",
        "",
        "| Candidate | Coordinates | Pixel sizes | Complete text | Bottom UI | Scene background | Long prompts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["candidates"]:
        lines.append(
            f"| {item['candidate_id']} | `{item['normalized_coordinates']}` | `{item['pixel_sizes']}` | "
            f"{item['contains_complete_text']} | {item['contains_bottom_ui']} | "
            f"{item['contains_large_scene_background']} | {item['covers_different_prompt_lengths']} |"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create human-review sheets for unapproved Prompt ROI candidates.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--session", action="append", help="Session id; repeatable")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "config" / "prompt_roi_candidates.yaml")
    parser.add_argument("--candidate", action="append", help="Limit to candidate id; repeatable")
    parser.add_argument("--samples-per-state", type=int, default=4)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        sessions = sorted(path for path in args.session_root.glob("session_*") if (path / "ground_truth.yaml").is_file())
    else:
        sessions = [args.session_root / session for session in args.session]
    candidates = load_roi_candidates(args.candidates)
    if args.candidate:
        allowed = set(args.candidate)
        candidates = [item for item in candidates if item.candidate_id in allowed]
    report = review_prompt_roi(sessions, candidates, args.report_dir, samples_per_state=args.samples_per_state)
    report_path = args.report_dir.parent / "roi_review_report.md"
    _write_report(report, report_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

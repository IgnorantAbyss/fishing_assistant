"""Generate manual-review artifacts for fixed pixel Prompt ROI candidates.

This tool deliberately does not approve an ROI, infer Prompt labels, run OCR, or
score candidates with a classifier. Global ground truth is used only to stratify
human-review frames and describe transition context.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_roi_candidates  # noqa: E402


DEFAULT_SESSION_ROOT = PROJECT_ROOT / "assets" / "replay" / "sessions"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "fishing_v2" / "roi_review"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "fishing_v2.yaml"
GLOBAL_STATES = ("IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET")
ALLOWED_TRANSITIONS = (
    ("IDLE", "WAITING"),
    ("WAITING", "READY"),
    ("READY", "HOOK"),
    ("HOOK", "PRESS"),
    ("HOOK", "GET"),
    ("PRESS", "GET"),
    ("PRESS", "IGNORE"),
    ("PRESS", "IDLE"),
    ("GET", "IDLE"),
    ("HOOK", "IDLE"),
)


@dataclass(frozen=True)
class ReviewSample:
    session_path: Path
    session_id: str
    frame_index: int
    frame_path: Path
    global_state: str
    distance_to_transition: int


@dataclass(frozen=True)
class TransitionWindow:
    session_id: str
    from_state: str
    to_state: str
    boundary_frame: int
    samples: tuple[ReviewSample, ...]

    @property
    def transition_id(self) -> str:
        return f"{self.from_state}_to_{self.to_state}"


@dataclass(frozen=True)
class SessionReviewData:
    path: Path
    frame_count: int
    image_format: str
    labels: dict[int, str]
    segments: tuple[dict[str, Any], ...]
    transition_boundaries: tuple[int, ...]


def _global_segments(path: Path, frame_count: int) -> tuple[tuple[dict[str, Any], ...], dict[int, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f"Global ground truth requires segments: {path}")
    segments: list[dict[str, Any]] = []
    labels: dict[int, str] = {}
    for item in raw:
        start, end, state = int(item["start"]), int(item["end"]), str(item["state"])
        segment = {"start": start, "end": end, "state": state}
        segments.append(segment)
        for frame in range(start, end + 1):
            if frame in labels:
                raise ValueError(f"Overlapping global ground truth: {path}")
            labels[frame] = state
    if set(labels) != set(range(1, frame_count + 1)):
        raise ValueError(f"Incomplete global ground truth: {path}")
    return tuple(segments), labels


def _load_session(path: Path) -> SessionReviewData:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest["frame_count"])
    image_format = str(manifest["image_format"])
    segments, labels = _global_segments(path / "ground_truth.yaml", frame_count)
    boundaries = tuple(int(item["start"]) for item in segments[1:])
    return SessionReviewData(path, frame_count, image_format, labels, segments, boundaries)


def _nearest_transition_distance(frame_index: int, boundaries: Sequence[int]) -> int:
    if not boundaries:
        return -1
    return min(min(abs(frame_index - boundary), abs(frame_index - (boundary - 1))) for boundary in boundaries)


def _uniform_sample(values: Sequence[int], limit: int) -> list[int]:
    if limit < 1:
        raise ValueError("samples_per_state must be positive")
    ordered = sorted(set(values))
    if len(ordered) <= limit:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, limit)
    return [ordered[round(float(position))] for position in positions]


def select_stable_interior(
    session: SessionReviewData, *, samples_per_state: int = 8
) -> dict[str, list[ReviewSample]]:
    """Select segment interiors without treating global state as a Prompt label."""
    by_state: dict[str, list[int]] = defaultdict(list)
    for segment in session.segments:
        state = str(segment["state"])
        if state not in GLOBAL_STATES:
            continue
        start, end = int(segment["start"]), int(segment["end"])
        length = end - start + 1
        trim = min(10, max(0, (length - 1) // 3))
        interior_start, interior_end = start + trim, end - trim
        by_state[state].extend(range(interior_start, interior_end + 1))
    selected: dict[str, list[ReviewSample]] = {}
    for state in GLOBAL_STATES:
        frames = _uniform_sample(by_state[state], samples_per_state) if by_state[state] else []
        selected[state] = [
            _sample(session, frame, state)
            for frame in frames
        ]
    return selected


def select_transition_windows(
    session: SessionReviewData, *, frames_before: int = 10, frames_after: int = 10
) -> list[TransitionWindow]:
    windows: list[TransitionWindow] = []
    for previous, current in zip(session.segments, session.segments[1:]):
        transition = (str(previous["state"]), str(current["state"]))
        if transition not in ALLOWED_TRANSITIONS:
            continue
        boundary = int(current["start"])
        indexes = range(
            max(1, boundary - frames_before),
            min(session.frame_count, boundary + frames_after - 1) + 1,
        )
        samples = tuple(_sample(session, frame, session.labels[frame]) for frame in indexes)
        windows.append(TransitionWindow(session.path.name, *transition, boundary, samples))
    return windows


def _sample(session: SessionReviewData, frame_index: int, state: str) -> ReviewSample:
    return ReviewSample(
        session.path,
        session.path.name,
        frame_index,
        session.path / "frames" / f"{frame_index:06d}.{session.image_format}",
        state,
        _nearest_transition_distance(frame_index, session.transition_boundaries),
    )


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


def with_metadata_margin(image: np.ndarray, lines: Sequence[str], *, margin_height: int = 66) -> np.ndarray:
    """Append labels below content so no black metadata overlay can hide a prompt."""
    output = np.zeros((image.shape[0] + margin_height, image.shape[1], 3), dtype=np.uint8)
    output[: image.shape[0]] = image
    y = image.shape[0] + 16
    for line in lines[:3]:
        cv2.putText(output, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
        y += 20
    return output


def _read_frame(sample: ReviewSample) -> np.ndarray:
    frame = cv2.imread(str(sample.frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(sample.frame_path)
    return frame


def _tile(sample: ReviewSample, candidate: PromptROICandidate, *, full_frame: bool) -> np.ndarray:
    frame = _read_frame(sample)
    left, top, right, bottom = candidate.pixel_bounds(frame.shape[1], frame.shape[0])
    pixel_text = f"px=[{left},{top},{right},{bottom}]"
    if full_frame:
        content = frame.copy()
        cv2.rectangle(content, (left, top), (right - 1, bottom - 1), (0, 255, 255), 4)
        fitted = _fit(content, 480, 270)
    else:
        # The crop content is untouched; all labels are appended in a separate margin.
        fitted = _fit(frame[top:bottom, left:right], 480, 128)
    return with_metadata_margin(fitted, (
        f"candidate={candidate.candidate_id} {pixel_text}",
        f"session={sample.session_id} frame={sample.frame_index}",
        f"global={sample.global_state} distance_to_transition={sample.distance_to_transition}",
    ))


def _contact_sheet(tiles: Sequence[np.ndarray], *, columns: int = 4) -> np.ndarray:
    if not tiles:
        return np.zeros((120, 480, 3), dtype=np.uint8)
    height, width = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    return sheet


def _write_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise OSError(f"Could not write ROI review image: {path}")
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_paged_tiles(directory: Path, prefix: str, tiles: Sequence[np.ndarray], *, page_size: int = 20) -> list[str]:
    paths: list[str] = []
    for page, offset in enumerate(range(0, len(tiles), page_size), start=1):
        sheet = _contact_sheet(tiles[offset:offset + page_size])
        paths.append(_write_image(directory / f"{prefix}__page_{page:02d}.jpg", sheet))
    return paths


def _candidate_report(candidate: PromptROICandidate, legacy_area: int) -> dict[str, Any]:
    reduction = 100.0 * (1.0 - candidate.area / legacy_area)
    return {
        "candidate_id": candidate.candidate_id,
        "pixel_coordinates": list(candidate.pixel),
        "normalized_coordinates_derived": [round(value, 6) for value in candidate.normalized],
        "width": candidate.width,
        "height": candidate.height,
        "area": candidate.area,
        "area_reduction_vs_legacy_percent": round(reduction, 2),
        "intended_content": candidate.intended_content,
        "note": candidate.note,
        "manual_review_required": True,
        "contains_complete_text": "manual_review_required",
        "contains_bottom_ui": "manual_review_required",
        "contains_large_scene_background": "manual_review_required",
        "covers_longest_prompt": "manual_review_required",
    }


def review_prompt_roi(
    session_paths: Sequence[Path],
    candidates: Sequence[PromptROICandidate],
    report_dir: Path,
    *,
    samples_per_state: int = 8,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one Prompt ROI candidate is required")
    sessions = [_load_session(path) for path in session_paths]
    if not sessions:
        raise ValueError("At least one review session is required")
    for session in sessions:
        if (candidates[0].reference_width, candidates[0].reference_height) != _manifest_screen_size(session.path):
            raise ValueError(f"Session violates fixed ROI environment: {session.path.name}")

    stable = {session.path.name: select_stable_interior(session, samples_per_state=samples_per_state) for session in sessions}
    transitions = {session.path.name: select_transition_windows(session) for session in sessions}
    directories = {
        name: report_dir / name
        for name in ("stable_interior", "transition_windows", "full_frame_with_roi", "cropped_roi")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    legacy = next((item for item in candidates if item.candidate_id == "legacy_reference"), candidates[0])
    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        details = _candidate_report(candidate, legacy.area)
        sheet_paths: dict[str, list[str]] = {name: [] for name in directories}
        for session in sessions:
            session_id = session.path.name
            overview_samples: list[ReviewSample] = []
            for state, samples in stable[session_id].items():
                if not samples:
                    continue
                overview_samples.extend(samples)
                tiles = [_tile(sample, candidate, full_frame=False) for sample in samples]
                path = directories["stable_interior"] / f"{candidate.candidate_id}__{session_id}__{state}.jpg"
                sheet_paths["stable_interior"].append(_write_image(path, _contact_sheet(tiles)))
            for window in transitions[session_id]:
                overview_samples.extend(
                    sample for sample in window.samples
                    if sample.frame_index in {window.boundary_frame - 1, window.boundary_frame}
                )
                tiles = [_tile(sample, candidate, full_frame=False) for sample in window.samples]
                prefix = f"{candidate.candidate_id}__{session_id}__{window.transition_id}__at_{window.boundary_frame:06d}"
                sheet_paths["transition_windows"].append(
                    _write_image(directories["transition_windows"] / f"{prefix}.jpg", _contact_sheet(tiles))
                )
            unique_overview = {sample.frame_index: sample for sample in overview_samples}
            ordered = [unique_overview[index] for index in sorted(unique_overview)]
            full_tiles = [_tile(sample, candidate, full_frame=True) for sample in ordered]
            crop_tiles = [_tile(sample, candidate, full_frame=False) for sample in ordered]
            sheet_paths["full_frame_with_roi"].extend(_write_paged_tiles(
                directories["full_frame_with_roi"], f"{candidate.candidate_id}__{session_id}", full_tiles
            ))
            sheet_paths["cropped_roi"].extend(_write_paged_tiles(
                directories["cropped_roi"], f"{candidate.candidate_id}__{session_id}", crop_tiles
            ))
        details["contact_sheets"] = sheet_paths
        candidate_reports.append(details)

    stable_counts = Counter()
    stable_session_state_counts: dict[str, dict[str, int]] = {}
    for session_id, groups in stable.items():
        stable_session_state_counts[session_id] = {state: len(samples) for state, samples in groups.items()}
        stable_counts.update({state: len(samples) for state, samples in groups.items()})
    transition_counts = Counter(
        window.transition_id for windows in transitions.values() for window in windows
    )
    transition_frame_counts = Counter()
    for windows in transitions.values():
        for window in windows:
            transition_frame_counts[window.transition_id] += len(window.samples)

    report: dict[str, Any] = {
        "fixed_environment": {
            "resolution": {"width": legacy.reference_width, "height": legacy.reference_height},
            "ui_scale": "fixed",
            "language": "zh-TW",
            "window_mode": "borderless",
            "prompt_position": "fixed",
            "supported_environment_only": True,
        },
        "roi_source_of_truth": "pixel",
        "normalized_coordinates_role": "derived_display_only",
        "roi_status": "unapproved",
        "manual_review_required": True,
        "automatic_selection_performed": False,
        "bright_mask_used_for_approval": False,
        "classifier_accuracy_used_for_selection": False,
        "sessions": [session.path.name for session in sessions],
        "excluded_trial_session": "session_20260709_192231",
        "candidates": candidate_reports,
        "stable_interior": {
            "max_samples_per_session_state": samples_per_state,
            "counts_by_global_state": dict(stable_counts),
            "total_samples": sum(stable_counts.values()),
            "counts_by_session_and_state": stable_session_state_counts,
        },
        "transition_windows": {
            "frames_before": 10,
            "frames_after": 10,
            "window_counts_by_transition": dict(transition_counts),
            "sample_counts_by_transition": dict(transition_frame_counts),
            "total_windows": sum(transition_counts.values()),
            "total_samples": sum(transition_frame_counts.values()),
        },
        "distinct_prompt_appearances": {
            "confirmed_count": None,
            "status": "not_yet_manually_confirmed",
            "note": "Contact sheets and inventory rows are review evidence, not human-confirmed Prompt classes.",
        },
        "manual_questions": [
            {"question": "Does each candidate contain the complete visible prompt?", "answer": "manual_review_required"},
            {"question": "Does any candidate include unrelated bottom UI?", "answer": "manual_review_required"},
            {"question": "Does any candidate include excessive scene background?", "answer": "manual_review_required"},
            {"question": "Which candidate preserves the longest prompt?", "answer": "manual_review_required"},
            {"question": "Are icon/key cues required Prompt content?", "answer": "manual_review_required"},
            {"question": "Does READY remain visible after the global READY-to-HOOK boundary?", "answer": "manual_review_required"},
            {"question": "Is a distinct PRESS instruction visible?", "answer": "manual_review_required"},
            {"question": "Is GET represented by a Prompt or by NO_PROMPT?", "answer": "manual_review_required"},
            {"question": "Which transition frames should be IGNORE?", "answer": "manual_review_required"},
            {"question": "Are there visually distinct prompts within one suggested observation kind?", "answer": "manual_review_required"},
        ],
        "provisional_review_order": [item.candidate_id for item in candidates],
        "final_recommendation": None,
    }
    _write_candidate_summary(report, report_dir / "roi_candidate_summary.md")
    _write_inventory(stable, transitions, report_dir.parent / "prompt_inventory_template.md")
    return report


def _manifest_screen_size(session_path: Path) -> tuple[int, int]:
    manifest = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
    raw = manifest.get("screen_size")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"Session manifest requires screen_size: {session_path}")
    return int(raw[0]), int(raw[1])


def _write_candidate_summary(report: dict[str, Any], destination: Path) -> None:
    lines = [
        "# Prompt ROI Candidate Summary",
        "",
        "Pixel coordinates are the source of truth at 2560x1440. Normalized values are derived display metadata.",
        "ROI status remains **unapproved**; every visual question requires manual review.",
        "",
        "| Candidate | Pixel ROI | Derived normalized ROI | Width | Height | Area | Reduction vs legacy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["candidates"]:
        lines.append(
            f"| {item['candidate_id']} | `{item['pixel_coordinates']}` | `{item['normalized_coordinates_derived']}` | "
            f"{item['width']} | {item['height']} | {item['area']} | {item['area_reduction_vs_legacy_percent']:.2f}% |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_inventory(
    stable: dict[str, dict[str, list[ReviewSample]]],
    transitions: dict[str, list[TransitionWindow]],
    destination: Path,
) -> None:
    lines = [
        "# Prompt Inventory Manual Review Template",
        "",
        "This inventory is intentionally unlabelled. Global state only selects review strata; it must not be copied into Prompt observation ground truth.",
        "Allowed suggestions after visual review: `IDLE_PROMPT`, `WAITING_PROMPT`, `READY_PROMPT`, `OTHER_PROMPT`, `NO_PROMPT`, `IGNORE`.",
        "Optional prompt-id examples: `IDLE_CAST`, `FISHING_IN_PROGRESS`, `FISH_BITE_SPACE`, `PRESS_SEQUENCE_INSTRUCTION`.",
        "",
        "Distinct Prompt appearance count: **not yet manually confirmed**.",
        "",
        "| session_id | frame_index_or_range | global_state | visible_prompt_description | suggested_observation_kind | suggested_prompt_id | candidate_roi_complete | transition_or_stable | manual_review_status | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for session_id in sorted(stable):
        for state in GLOBAL_STATES:
            samples = stable[session_id].get(state, [])
            if not samples:
                continue
            indexes = ", ".join(str(sample.frame_index) for sample in samples)
            lines.append(
                f"| {session_id} | {indexes} | {state} |  |  |  | manual_review_required | stable | manual_review_required | Global state is context only. |"
            )
        for window in transitions[session_id]:
            start, end = window.samples[0].frame_index, window.samples[-1].frame_index
            lines.append(
                f"| {session_id} | {start}-{end} | {window.from_state}→{window.to_state} |  |  |  | manual_review_required | transition | manual_review_required | Boundary frame {window.boundary_frame}; review each side. |"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(report: dict[str, Any], markdown_path: Path, json_path: Path) -> None:
    lines = [
        "# Prompt ROI Review Report",
        "",
        "- Fixed environment: **2560x1440**, fixed UI scale, zh-TW, borderless, fixed Prompt position.",
        "- ROI source of truth: **pixel coordinates**; normalized coordinates are derived display metadata only.",
        "- ROI status: **unapproved**.",
        "- Manual review required: **true**.",
        "- Automatic selection / bright-mask approval / classifier-accuracy selection: **false / false / false**.",
        f"- Sessions: {', '.join(report['sessions'])}",
        f"- Trial session excluded: `{report['excluded_trial_session']}`",
        "- Distinct Prompt appearances: **not yet manually confirmed**.",
        "",
        "## Candidates",
        "",
        "| Candidate | Pixel | Derived normalized | Size | Area | Reduction vs legacy |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in report["candidates"]:
        lines.append(
            f"| {item['candidate_id']} | `{item['pixel_coordinates']}` | `{item['normalized_coordinates_derived']}` | "
            f"{item['width']}x{item['height']} | {item['area']} | {item['area_reduction_vs_legacy_percent']:.2f}% |"
        )
    lines.extend([
        "",
        "## Sampling",
        "",
        f"- Stable interior total: {report['stable_interior']['total_samples']}",
        f"- Stable counts by global state: `{report['stable_interior']['counts_by_global_state']}`",
        f"- Transition windows total: {report['transition_windows']['total_windows']}",
        f"- Transition samples total: {report['transition_windows']['total_samples']}",
        f"- Transition window counts: `{report['transition_windows']['window_counts_by_transition']}`",
        "- Large visual sheets: `reports/fishing_v2/roi_review/` (ignored by Git).",
        "",
        "## Manual questions",
        "",
    ])
    lines.extend(f"- {item['question']} — **{item['answer']}**" for item in report["manual_questions"])
    lines.extend([
        "",
        "## Provisional review order",
        "",
        ", ".join(f"`{item}`" for item in report["provisional_review_order"]),
        "",
        "This is a review order, not a final recommendation or approval.",
    ])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _configured_session_ids(config_path: Path) -> tuple[list[str], list[str]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    review = config.get("prompt", {}).get("roi_review", {})
    return list(review.get("session_ids", [])), list(review.get("excluded_trial_sessions", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create human-review sheets for unapproved fixed-pixel Prompt ROI candidates.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Use the seven configured review sessions; excludes trial sessions.")
    selection.add_argument("--session", action="append", help="Session id; repeatable")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "config" / "prompt_roi_candidates.yaml")
    parser.add_argument("--candidate", action="append", help="Limit to candidate id; repeatable")
    parser.add_argument("--samples-per-state", type=int, default=8)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configured, excluded = _configured_session_ids(args.config)
    if args.all:
        selected_ids = configured
    else:
        selected_ids = list(args.session)
    forbidden = sorted(set(selected_ids) & set(excluded))
    if forbidden:
        raise ValueError(f"Trial sessions are excluded from Prompt ROI review: {', '.join(forbidden)}")
    sessions = [args.session_root / session_id for session_id in selected_ids]
    missing = [path.name for path in sessions if not (path / "ground_truth.yaml").is_file()]
    if missing:
        raise FileNotFoundError(f"Review sessions missing ground truth: {', '.join(missing)}")
    candidates = load_roi_candidates(args.candidates)
    if args.candidate:
        allowed = set(args.candidate)
        candidates = [item for item in candidates if item.candidate_id in allowed]
        if not candidates:
            raise ValueError("No requested ROI candidate id was found")
    report = review_prompt_roi(sessions, candidates, args.report_dir, samples_per_state=args.samples_per_state)
    markdown_path = args.report_dir.parent / "roi_review_report.md"
    json_path = args.report_dir.parent / "roi_review_report.json"
    _write_report(report, markdown_path, json_path)
    print(json.dumps({
        "report": str(markdown_path),
        "json": str(json_path),
        "inventory": str(args.report_dir.parent / "prompt_inventory_template.md"),
        "roi_status": report["roi_status"],
        "manual_review_required": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

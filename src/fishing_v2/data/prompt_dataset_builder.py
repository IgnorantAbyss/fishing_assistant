"""Prompt Observation dataset builder with explicit annotation/ROI lineage."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.fishing_v2.data.dataset_lineage import BUILDER_VERSION, sha256_file, write_dataset_lineage
from src.fishing_v2.data.prompt_annotation import (
    PromptAnnotationKind,
    load_prompt_annotations,
    prompt_boundary_frames,
)
from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_approved_prompt_roi


MANIFEST_FIELDS = (
    "session_id", "frame_index", "source_frame", "crop_path",
    "prompt_observation", "prompt_id", "global_state", "is_boundary_global",
    "is_boundary_prompt", "split", "usage",
    "source_prompt_ground_truth_sha256", "source_global_ground_truth_sha256",
    "source_frame_sha256", "roi_config_sha256", "dataset_version",
)


@dataclass(frozen=True)
class PromptDatasetBuildResult:
    dry_run: bool
    row_count: int
    output_root: Path
    rows: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


def _segments_to_labels(path: Path, frame_count: int) -> dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        raise ValueError(f"Global ground truth requires segments: {path}")
    labels: dict[int, str] = {}
    for item in segments:
        for frame in range(int(item["start"]), int(item["end"]) + 1):
            labels[frame] = str(item["state"])
    if set(labels) != set(range(1, frame_count + 1)):
        raise ValueError(f"Global ground truth does not cover session: {path.parent.name}")
    return labels


def _boundaries(labels: dict[int, Any]) -> set[int]:
    boundaries: set[int] = set()
    previous: Any = None
    for frame, label in sorted(labels.items()):
        if previous is not None and label != previous:
            boundaries.update({frame - 1, frame})
        previous = label
    return boundaries


def _split_mapping(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for session_id in data.get(split, []):
            result[str(session_id)] = split
    return result


def build_prompt_dataset(
    *,
    config_path: str | Path,
    split_path: str | Path,
    session_root: str | Path,
    output_root: str | Path,
    dry_run: bool,
    boundary_policy: str = "stable_only",
    dry_run_candidate: PromptROICandidate | None = None,
) -> PromptDatasetBuildResult:
    if boundary_policy not in {"stable_only", "include_prompt_boundary"}:
        raise ValueError("boundary_policy must be stable_only or include_prompt_boundary")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    approved_roi = load_approved_prompt_roi(config_path)
    if approved_roi is None and not dry_run:
        raise PermissionError("Prompt ROI is unapproved; only dry-run dataset planning is allowed")
    roi = approved_roi or dry_run_candidate
    if roi is None:
        return PromptDatasetBuildResult(
            True, 0, Path(output_root), (),
            ("Prompt ROI is unapproved and no dry-run candidate was supplied.",),
        )
    root = Path(session_root)
    output = Path(output_root)
    if output.resolve() == Path(config["data"]["legacy_dataset_root"]).resolve():
        raise ValueError("Refusing to overwrite frozen datasets/fishing_v2")
    development_sessions = set(config["data"].get("development_sessions", []))
    split_by_session = _split_mapping(Path(split_path))
    rows: list[dict[str, Any]] = []
    source_sessions: list[dict[str, Any]] = []
    for session_path in sorted(path for path in root.glob("session_*") if path.is_dir()):
        prompt_gt = session_path / "prompt_ground_truth.yaml"
        global_gt = session_path / "ground_truth.yaml"
        manifest_path = session_path / "manifest.json"
        if not (prompt_gt.is_file() and global_gt.is_file() and manifest_path.is_file()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_count = int(manifest["frame_count"])
        image_format = str(manifest["image_format"])
        prompt_annotations = load_prompt_annotations(prompt_gt, frame_count)
        prompt_labels = {frame: annotation.observation for frame, annotation in prompt_annotations.items()}
        global_labels = _segments_to_labels(global_gt, frame_count)
        prompt_boundaries = prompt_boundary_frames(prompt_labels)
        global_boundaries = _boundaries(global_labels)
        source_sessions.append({
            "session_id": session_path.name,
            "prompt_annotation_sha256": sha256_file(prompt_gt),
            "global_annotation_sha256": sha256_file(global_gt),
        })
        for frame_index in range(1, frame_count + 1):
            prompt_label = prompt_labels[frame_index]
            prompt_annotation = prompt_annotations[frame_index]
            if prompt_label == PromptAnnotationKind.IGNORE:
                continue
            source_frame = session_path / "frames" / f"{frame_index:06d}.{image_format}"
            if not source_frame.is_file():
                raise FileNotFoundError(source_frame)
            crop_relative = Path("crops") / f"{session_path.name}_{frame_index:06d}_{prompt_label.value.lower()}.jpg"
            usage = "development_diagnostic" if session_path.name in development_sessions else "development"
            row = {
                "session_id": session_path.name,
                "frame_index": frame_index,
                "source_frame": source_frame.as_posix(),
                "crop_path": crop_relative.as_posix(),
                "prompt_observation": prompt_label.value,
                "prompt_id": prompt_annotation.prompt_id,
                "global_state": global_labels[frame_index],
                "is_boundary_global": frame_index in global_boundaries,
                "is_boundary_prompt": frame_index in prompt_boundaries,
                "split": split_by_session.get(session_path.name, "unassigned"),
                "usage": usage,
                "source_prompt_ground_truth_sha256": sha256_file(prompt_gt),
                "source_global_ground_truth_sha256": sha256_file(global_gt),
                "source_frame_sha256": sha256_file(source_frame),
                "roi_config_sha256": sha256_file(config_path),
                "dataset_version": str(config["data"]["dataset_version"]),
            }
            rows.append(row)
            if not dry_run:
                frame = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
                if frame is None:
                    raise FileNotFoundError(source_frame)
                left, top, right, bottom = roi.pixel_bounds(frame.shape[1], frame.shape[0])
                destination = output / crop_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), frame[top:bottom, left:right]):
                    raise OSError(f"Could not write crop: {destination}")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        lineage = {
            "dataset_version": config["data"]["dataset_version"],
            "source_sessions": source_sessions,
            "roi_config_sha256": sha256_file(config_path),
            "builder_version": BUILDER_VERSION,
            "old_dataset_reference": config["data"]["legacy_dataset_root"],
            "excluded_sessions": [],
            "development_sessions": sorted(development_sessions),
            "final_test_eligibility": config["data"]["final_test"],
            "boundary_policy": boundary_policy,
            "label_source": "prompt_ground_truth.yaml only",
        }
        write_dataset_lineage(output / "dataset_lineage.json", lineage)
    return PromptDatasetBuildResult(dry_run, len(rows), output, tuple(rows), ())

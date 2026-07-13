"""Build the fixed-ROI Prompt Observation v1 evaluation dataset.

Labels come exclusively from human-authored ``prompt_ground_truth.yaml``.  The
global fishing state is deliberately not loaded by this module.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2

from src.fishing_v2.data.prompt_annotation import (
    FINAL_PROMPT_ANNOTATION_KINDS,
    PromptAnnotationKind,
    load_prompt_ground_truth,
)
from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_approved_prompt_roi


FORMAL_SESSION_IDS = (
    "session_20260709_192315",
    "session_20260710_061220",
    "session_20260710_123210",
    "session_20260710_124419",
    "session_20260710_125441",
    "session_20260710_130308",
    "session_20260710_131254",
)
TRIAL_SESSION_ID = "session_20260709_192231"
MANIFEST_FIELDS = (
    "session_id",
    "frame_index",
    "label",
    "source_frame_path",
    "roi_path",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "image_sha256",
)


@dataclass(frozen=True)
class PromptObservationDatasetResult:
    output_root: Path
    manifest_path: Path
    rows: tuple[dict[str, str], ...]
    summary: Mapping[str, Any]


def _manifest(path: Path) -> tuple[int, str, tuple[int, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    frame_count = raw.get("frame_count")
    image_format = raw.get("image_format")
    screen_size = raw.get("screen_size")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        raise ValueError(f"Invalid replay frame_count: {path}")
    if not isinstance(image_format, str) or not image_format:
        raise ValueError(f"Invalid replay image_format: {path}")
    if not (
        isinstance(screen_size, list)
        and len(screen_size) == 2
        and all(isinstance(item, int) for item in screen_size)
    ):
        raise ValueError(f"Invalid replay screen_size: {path}")
    return frame_count, image_format, (screen_size[0], screen_size[1])


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _encode_png(image: Any) -> bytes:
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise OSError("Could not encode Prompt ROI")
    return encoded.tobytes()


def _write_manifest(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_prompt_observation_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != MANIFEST_FIELDS:
        raise ValueError(f"Invalid Prompt Observation manifest schema: {path}")
    return rows


def validate_cross_label_hashes(rows: Sequence[Mapping[str, str]]) -> None:
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    locations: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        label = str(row["label"])
        digest = str(row["image_sha256"])
        if label != PromptAnnotationKind.IGNORE.value:
            labels_by_hash[digest].add(label)
            locations[digest].append(f"{row['session_id']}:{row['frame_index']}={label}")
    conflicts = {
        digest: locations[digest]
        for digest, labels in labels_by_hash.items()
        if len(labels) > 1
    }
    if conflicts:
        details = "; ".join(
            f"{digest}: {', '.join(items)}" for digest, items in sorted(conflicts.items())
        )
        raise ValueError(f"Cross-label duplicate Prompt ROI conflict: {details}")


def build_prompt_observation_dataset(
    *,
    config_path: str | Path,
    session_root: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    session_ids: Sequence[str] = FORMAL_SESSION_IDS,
    materialize_crops: bool = True,
) -> PromptObservationDatasetResult:
    config_path = Path(config_path)
    session_root = Path(session_root)
    output_root = Path(output_root)
    project_root = Path(project_root)
    if tuple(session_ids) != FORMAL_SESSION_IDS:
        raise ValueError("Prompt Observation v1 requires exactly the seven formal sessions")
    if TRIAL_SESSION_ID in session_ids:
        raise ValueError("Trial replay must not enter Prompt Observation v1")
    roi = load_approved_prompt_roi(config_path)
    if roi is None or roi.pixel != (940, 36, 1620, 100):
        raise ValueError("Prompt Observation v1 requires approved pixel ROI [940, 36, 1620, 100]")

    rows: list[dict[str, str]] = []
    session_counts: dict[str, Counter[str]] = {}
    hash_locations: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for session_id in session_ids:
        session_path = session_root / session_id
        frame_count, image_format, screen_size = _manifest(session_path / "manifest.json")
        if frame_count != 600:
            raise ValueError(f"Formal Prompt annotation must cover 600 frames: {session_id}")
        if screen_size != (roi.reference_width, roi.reference_height):
            raise ValueError(f"Unsupported Prompt frame size for {session_id}: {screen_size}")
        labels = load_prompt_ground_truth(session_path / "prompt_ground_truth.yaml", frame_count)
        if set(labels) != set(range(1, frame_count + 1)):
            raise ValueError(f"Prompt annotation coverage is incomplete: {session_id}")
        legal = {item.value for item in FINAL_PROMPT_ANNOTATION_KINDS}
        if {item.value for item in labels.values()} - legal:
            raise ValueError(f"Prompt annotation uses an illegal label: {session_id}")
        counts: Counter[str] = Counter()
        for frame_index in range(1, frame_count + 1):
            source = session_path / "frames" / f"{frame_index:06d}.{image_format}"
            frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(source)
            bounds = roi.pixel_bounds(frame.shape[1], frame.shape[0])
            x1, y1, x2, y2 = bounds
            crop = frame[y1:y2, x1:x2]
            if crop.shape[:2] != (roi.height, roi.width):
                raise ValueError(f"Invalid Prompt crop size: {session_id}:{frame_index}")
            encoded = _encode_png(crop)
            digest = hashlib.sha256(encoded).hexdigest()
            label = labels[frame_index].value
            crop_relative = Path("crops") / session_id / f"{frame_index:06d}.png"
            if materialize_crops:
                destination = output_root / crop_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded)
            row = {
                "session_id": session_id,
                "frame_index": str(frame_index),
                "label": label,
                "source_frame_path": _relative(source, project_root),
                "roi_path": crop_relative.as_posix(),
                "roi_x1": str(x1),
                "roi_y1": str(y1),
                "roi_x2": str(x2),
                "roi_y2": str(y2),
                "image_sha256": digest,
            }
            rows.append(row)
            counts[label] += 1
            hash_locations[digest].append((session_id, frame_index, label))
        session_counts[session_id] = counts

    validate_cross_label_hashes(rows)
    manifest_path = output_root / "manifest.csv"
    _write_manifest(manifest_path, rows)
    label_counts = Counter(row["label"] for row in rows)
    summary: dict[str, Any] = {
        "version": 1,
        "dataset": "prompt_observation_v1",
        "label_source": "prompt_ground_truth.yaml only",
        "global_ground_truth_used": False,
        "approved_roi": list(roi.pixel),
        "reference_resolution": [roi.reference_width, roi.reference_height],
        "sessions": list(session_ids),
        "excluded_sessions": [TRIAL_SESSION_ID],
        "frame_count": len(rows),
        "session_label_counts": {
            session_id: {label: session_counts[session_id].get(label, 0) for label in sorted(label_counts)}
            for session_id in session_ids
        },
        "label_counts": {label: label_counts[label] for label in sorted(label_counts)},
        "unique_image_hashes": len(hash_locations),
        "duplicate_image_frames": len(rows) - len(hash_locations),
        "cross_non_ignore_label_conflicts": 0,
        "crop_size": [roi.width, roi.height],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return PromptObservationDatasetResult(output_root, manifest_path, tuple(rows), summary)


def resolve_roi_path(row: Mapping[str, str], dataset_root: str | Path) -> Path:
    return Path(dataset_root) / str(row["roi_path"])

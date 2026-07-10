"""CSV manifest handling and deterministic duplicate protection."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_FIELDS = (
    "session_id",
    "frame_index",
    "source_frame",
    "crop_path",
    "label",
    "original_state",
    "split",
    "selected_for_training",
    "excluded_reason",
    "is_boundary",
    "width",
    "height",
    "sha256",
)


class ManifestConflictError(ValueError):
    """Raised when identical visual bytes have conflicting labels."""


@dataclass(frozen=True)
class ManifestRow:
    session_id: str
    frame_index: int
    source_frame: str
    crop_path: str
    label: str
    original_state: str
    split: str
    selected_for_training: bool
    excluded_reason: str
    is_boundary: bool
    width: int
    height: int
    sha256: str

    def to_csv(self) -> dict[str, str | int]:
        data = asdict(self)
        data["is_boundary"] = "true" if self.is_boundary else "false"
        data["selected_for_training"] = "true" if self.selected_for_training else "false"
        return data


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Unexpected dataset manifest fields: {manifest_path}")
        return [
            ManifestRow(
                session_id=row["session_id"],
                frame_index=int(row["frame_index"]),
                source_frame=row["source_frame"],
                crop_path=row["crop_path"],
                label=row["label"],
                original_state=row["original_state"],
                split=row["split"],
                selected_for_training=row["selected_for_training"].lower() == "true",
                excluded_reason=row["excluded_reason"],
                is_boundary=row["is_boundary"].lower() == "true",
                width=int(row["width"]),
                height=int(row["height"]),
                sha256=row["sha256"],
            )
            for row in reader
        ]


def write_manifest(path: str | Path, rows: Iterable[ManifestRow]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row.session_id, row.frame_index, row.label))
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(row.to_csv() for row in ordered)
    temporary.replace(manifest_path)


class ManifestIndex:
    def __init__(self, rows: Iterable[ManifestRow] = ()) -> None:
        self.rows = list(rows)
        self.source_keys = {
            (row.session_id, row.frame_index, row.label) for row in self.rows
        }
        self.hash_labels: dict[str, set[str]] = {}
        for row in self.rows:
            self.hash_labels.setdefault(row.sha256, set()).add(row.label)

    def classify(self, row: ManifestRow) -> str:
        source_key = (row.session_id, row.frame_index, row.label)
        if source_key in self.source_keys:
            return "existing_source"
        labels = self.hash_labels.get(row.sha256, set())
        if labels and row.label not in labels:
            raise ManifestConflictError(
                f"Crop {row.sha256} has conflicting labels: {sorted(labels)} vs {row.label}"
            )
        if row.label in labels:
            return "duplicate_crop"
        return "new"

    def add(self, row: ManifestRow) -> None:
        if self.classify(row) != "new":
            raise ValueError("Only new manifest rows may be added")
        self.rows.append(row)
        self.source_keys.add((row.session_id, row.frame_index, row.label))
        self.hash_labels.setdefault(row.sha256, set()).add(row.label)

"""Dataset manifest, crop, and session-split validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from src.dataset.manifest import ManifestRow, read_manifest


PROMPT_LABELS = {"IDLE", "WAITING", "READY", "NONE"}
SPECIAL_LABELS = {"HOOK", "PRESS", "GET", "NONE"}
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SessionSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    unassigned: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "unassigned": list(self.unassigned),
        }

    def split_for(self, session_id: str) -> str:
        for name in (*SPLIT_NAMES, "unassigned"):
            if session_id in getattr(self, name):
                return name
        return "unassigned"


@dataclass(frozen=True)
class DatasetValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    prompt_rows: tuple[ManifestRow, ...]
    special_rows: tuple[ManifestRow, ...]


def validate_session_split(split: SessionSplit) -> None:
    seen: dict[str, str] = {}
    for name in (*SPLIT_NAMES, "unassigned"):
        sessions = getattr(split, name)
        if len(sessions) != len(set(sessions)):
            raise ValueError(f"Duplicate session within {name} split")
        for session_id in sessions:
            previous = seen.get(session_id)
            if previous is not None:
                raise ValueError(
                    f"Session {session_id} appears in both {previous} and {name} splits"
                )
            seen[session_id] = name


def load_session_split(path: str | Path, session_ids: Iterable[str]) -> SessionSplit:
    split_path = Path(path)
    known = tuple(sorted(set(session_ids)))
    if not split_path.is_file():
        return SessionSplit((), (), (), known)
    data = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Session split root must be a mapping: {split_path}")

    parsed: dict[str, tuple[str, ...]] = {}
    for name in (*SPLIT_NAMES, "unassigned"):
        value = data.get(name, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"session_split.{name} must be a list of session ids")
        parsed[name] = tuple(value)
    assigned = set().union(*(set(parsed[name]) for name in SPLIT_NAMES))
    parsed["unassigned"] = tuple(sorted((set(parsed["unassigned"]) | set(known)) - assigned))
    split = SessionSplit(
        parsed["train"], parsed["validation"], parsed["test"], parsed["unassigned"]
    )
    validate_session_split(split)
    return split


def write_session_split(path: str | Path, split: SessionSplit) -> None:
    validate_session_split(split)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(split.as_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _validate_rows(
    rows: list[ManifestRow], output_dir: Path, valid_labels: set[str], dataset_name: str
) -> list[str]:
    errors: list[str] = []
    source_keys: set[tuple[str, int, str]] = set()
    hash_labels: dict[str, set[str]] = {}
    for row in rows:
        key = (row.session_id, row.frame_index, row.label)
        if key in source_keys:
            errors.append(f"{dataset_name}: duplicate source key {key}")
        source_keys.add(key)
        if row.label not in valid_labels:
            errors.append(f"{dataset_name}: invalid label {row.label}")
        labels = hash_labels.setdefault(row.sha256, set())
        labels.add(row.label)
        if len(labels) > 1:
            errors.append(f"{dataset_name}: sha256 {row.sha256} has conflicting labels")
        crop_path = output_dir / row.crop_path
        if not crop_path.is_file():
            errors.append(f"{dataset_name}: missing crop {row.crop_path}")
            continue
        actual_hash = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        if actual_hash != row.sha256:
            errors.append(f"{dataset_name}: sha256 mismatch {row.crop_path}")
    return errors


def validate_dataset(
    output_dir: str | Path, *, required_session: str | None = None
) -> DatasetValidationResult:
    root = Path(output_dir)
    prompt_path = root / "prompt" / "manifest.csv"
    special_path = root / "special" / "manifest.csv"
    errors: list[str] = []
    try:
        prompt_rows = read_manifest(prompt_path)
    except (OSError, ValueError) as exc:
        prompt_rows = []
        errors.append(f"prompt manifest: {exc}")
    try:
        special_rows = read_manifest(special_path)
    except (OSError, ValueError) as exc:
        special_rows = []
        errors.append(f"special manifest: {exc}")
    if not prompt_path.is_file():
        errors.append("prompt manifest is missing")
    if not special_path.is_file():
        errors.append("special manifest is missing")
    errors.extend(_validate_rows(prompt_rows, root, PROMPT_LABELS, "prompt"))
    errors.extend(_validate_rows(special_rows, root, SPECIAL_LABELS, "special"))
    if required_session is not None:
        if not any(row.session_id == required_session for row in prompt_rows):
            errors.append(f"prompt dataset has no rows for {required_session}")
        if not any(row.session_id == required_session for row in special_rows):
            errors.append(f"special dataset has no rows for {required_session}")

    warnings: list[str] = []
    all_rows = [*prompt_rows, *special_rows]
    session_ids = {row.session_id for row in all_rows}
    try:
        split = load_session_split(root / "session_split.yaml", session_ids)
        for name in SPLIT_NAMES:
            if not getattr(split, name):
                warnings.append(f"{name} has no assigned sessions")
    except ValueError as exc:
        errors.append(str(exc))
    for dataset_name, rows, labels in (
        ("prompt", prompt_rows, PROMPT_LABELS),
        ("special", special_rows, SPECIAL_LABELS),
    ):
        present = {row.label for row in rows}
        for missing in sorted(labels - present):
            warnings.append(f"{dataset_name} label {missing} has no samples")
    return DatasetValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        prompt_rows=tuple(prompt_rows),
        special_rows=tuple(special_rows),
    )

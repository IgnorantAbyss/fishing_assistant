from pathlib import Path

import pytest

from src.dataset.manifest import (
    ManifestConflictError,
    ManifestIndex,
    ManifestRow,
    read_manifest,
    write_manifest,
)


def _row(
    *,
    frame: int = 1,
    label: str = "WAITING",
    digest: str = "a" * 64,
    selected: bool = True,
    excluded_reason: str = "",
) -> ManifestRow:
    return ManifestRow(
        session_id="session_test",
        frame_index=frame,
        source_frame=f"frames/{frame:06d}.jpg",
        crop_path=f"prompt/crops/{frame:06d}.jpg",
        label=label,
        original_state=label,
        split="train",
        selected_for_training=selected,
        excluded_reason=excluded_reason,
        is_boundary=False,
        width=320,
        height=64,
        sha256=digest,
    )


def test_manifest_round_trip_preserves_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    row = _row()

    write_manifest(path, [row])

    assert read_manifest(path) == [row]


def test_same_source_or_same_crop_is_not_added_twice() -> None:
    first = _row()
    index = ManifestIndex([first])

    assert index.classify(first) == "existing_source"
    assert index.classify(_row(frame=2)) == "duplicate_crop"
    assert len(index.rows) == 1


def test_identical_crop_with_different_label_is_not_silently_merged() -> None:
    index = ManifestIndex([_row()])

    with pytest.raises(ManifestConflictError, match="conflicting labels"):
        index.classify(_row(frame=2, label="READY"))


def test_manifest_preserves_majority_exclusion_instead_of_dropping_row(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    excluded = _row(selected=False, excluded_reason="majority_downsample")

    write_manifest(path, [excluded])
    loaded = read_manifest(path)

    assert loaded == [excluded]
    assert loaded[0].split == "train"
    assert loaded[0].selected_for_training is False
    assert loaded[0].excluded_reason == "majority_downsample"

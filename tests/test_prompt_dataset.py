from pathlib import Path

from PIL import Image
import pytest

from src.dataset.manifest import ManifestRow, write_manifest
from src.ml.prompt_dataset import CLASS_NAMES, CLASS_TO_INDEX, PromptManifestDataset


def _row(
    split: str,
    label: str,
    frame: int,
    *,
    selected: bool,
    session: str | None = None,
) -> ManifestRow:
    return ManifestRow(
        session_id=session or f"session_{split}",
        frame_index=frame,
        source_frame=f"source/{frame:06d}.jpg",
        crop_path=f"prompt/crops/{split}_{frame:06d}.jpg",
        label=label,
        original_state=label,
        split=split,
        selected_for_training=selected,
        excluded_reason="" if selected else ("evaluation_only" if split != "train" else "majority_downsample"),
        is_boundary=False,
        width=32,
        height=12,
        sha256=str(frame).zfill(64),
    )


@pytest.fixture
def prompt_manifest(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _row("train", "IDLE", 1, selected=True),
        _row("train", "WAITING", 2, selected=False),
        _row("validation", "READY", 3, selected=False),
        _row("validation", "NONE", 4, selected=False),
        _row("test", "IDLE", 5, selected=False),
    ]
    for row in rows:
        path = tmp_path / row.crop_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 12), (row.frame_index * 20, 10, 30)).save(path)
    manifest = tmp_path / "prompt" / "manifest.csv"
    write_manifest(manifest, rows)
    return manifest, tmp_path


def test_train_loader_uses_only_selected_rows(prompt_manifest) -> None:
    manifest, root = prompt_manifest
    dataset = PromptManifestDataset(manifest, root, "train")
    assert len(dataset) == 1
    assert dataset.rows[0].label == "IDLE"


def test_validation_loader_uses_all_split_rows(prompt_manifest) -> None:
    manifest, root = prompt_manifest
    dataset = PromptManifestDataset(manifest, root, "validation")
    assert [row.label for row in dataset.rows] == ["READY", "NONE"]


def test_test_loader_keeps_evaluation_only_row(prompt_manifest) -> None:
    manifest, root = prompt_manifest
    dataset = PromptManifestDataset(manifest, root, "test")
    assert len(dataset) == 1
    assert dataset.rows[0].selected_for_training is False


def test_class_mapping_order_is_fixed() -> None:
    assert CLASS_NAMES == ("IDLE", "WAITING", "READY", "NONE")
    assert CLASS_TO_INDEX == {"IDLE": 0, "WAITING": 1, "READY": 2, "NONE": 3}


def test_dataset_returns_metadata_and_rgb_image(prompt_manifest) -> None:
    manifest, root = prompt_manifest
    image, label, metadata = PromptManifestDataset(manifest, root, "train")[0]
    assert image.mode == "RGB"
    assert label == 0
    assert metadata["session_id"] == "session_train"
    assert metadata["frame_index"] == 1


def test_dataset_class_counts_follow_loaded_rows(prompt_manifest) -> None:
    manifest, root = prompt_manifest
    counts = PromptManifestDataset(manifest, root, "validation").class_counts
    assert counts == {"IDLE": 0, "WAITING": 0, "READY": 1, "NONE": 1}


def test_unsupported_split_is_rejected(prompt_manifest) -> None:
    manifest, root = prompt_manifest
    with pytest.raises(ValueError, match="Unsupported"):
        PromptManifestDataset(manifest, root, "trial")


def test_invalid_manifest_label_is_rejected(tmp_path: Path) -> None:
    row = _row("train", "HOOK", 1, selected=True)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [row])
    with pytest.raises(ValueError, match="invalid labels"):
        PromptManifestDataset(manifest, tmp_path, "train")

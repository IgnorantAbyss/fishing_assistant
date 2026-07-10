"""Manifest-backed prompt dataset with strict split isolation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from src.dataset.manifest import ManifestRow, read_manifest


CLASS_NAMES: tuple[str, ...] = ("IDLE", "WAITING", "READY", "NONE")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
EVALUATION_SPLITS = {"validation", "test"}


def select_prompt_rows(rows: list[ManifestRow], split: str) -> list[ManifestRow]:
    """Apply the one permitted selection rule for a requested split."""
    if split not in {"train", *EVALUATION_SPLITS}:
        raise ValueError(f"Unsupported prompt dataset split: {split}")
    selected = [row for row in rows if row.split == split]
    if split == "train":
        selected = [row for row in selected if row.selected_for_training]
    return selected


class PromptManifestDataset(Dataset):
    """Load prompt crops without allowing evaluation rows into training."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        rows = read_manifest(self.manifest_path)
        invalid = sorted({row.label for row in rows} - set(CLASS_NAMES))
        if invalid:
            raise ValueError(f"Prompt manifest contains invalid labels: {invalid}")
        self.rows = select_prompt_rows(rows, split)
        if not self.rows:
            raise ValueError(f"Prompt dataset split has no samples: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        crop_path = self.dataset_root / row.crop_path
        with Image.open(crop_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        metadata = {
            "session_id": row.session_id,
            "frame_index": row.frame_index,
            "source_frame": row.source_frame,
            "crop_path": str(crop_path),
            "label": row.label,
            "original_state": row.original_state,
            "is_boundary": row.is_boundary,
        }
        return image, CLASS_TO_INDEX[row.label], metadata

    @property
    def class_counts(self) -> dict[str, int]:
        counts = Counter(row.label for row in self.rows)
        return {name: counts.get(name, 0) for name in CLASS_NAMES}

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.session_id for row in self.rows}))

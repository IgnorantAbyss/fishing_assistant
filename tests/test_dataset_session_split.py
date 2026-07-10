from pathlib import Path

import pytest
import yaml

from src.dataset.validation import SessionSplit, load_session_split, validate_session_split


def test_missing_split_leaves_every_session_unassigned(tmp_path: Path) -> None:
    split = load_session_split(
        tmp_path / "session_split.yaml", ["session_a", "session_b"]
    )

    assert split.train == ()
    assert split.validation == ()
    assert split.test == ()
    assert split.unassigned == ("session_a", "session_b")


def test_same_session_cannot_cross_train_validation_or_test(tmp_path: Path) -> None:
    path = tmp_path / "session_split.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": ["session_a"],
                "validation": [],
                "test": ["session_a"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both train and test"):
        load_session_split(path, ["session_a"])


def test_manual_non_overlapping_split_is_valid() -> None:
    validate_session_split(
        SessionSplit(("session_train",), ("session_validation",), ("session_test",), ())
    )

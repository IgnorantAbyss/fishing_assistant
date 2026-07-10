from pathlib import Path

from src.ml.prompt_dataset import PromptManifestDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "fishing_v2"
MANIFEST = DATASET_ROOT / "prompt" / "manifest.csv"


def _dataset(split: str) -> PromptManifestDataset:
    return PromptManifestDataset(MANIFEST, DATASET_ROOT, split)


def test_real_dataset_sessions_do_not_leak_between_splits() -> None:
    train = set(_dataset("train").session_ids)
    validation = set(_dataset("validation").session_ids)
    test = set(_dataset("test").session_ids)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_trial_session_is_absent_from_every_loader() -> None:
    all_sessions = set().union(*(_dataset(split).session_ids for split in ("train", "validation", "test")))
    assert "session_20260709_192231" not in all_sessions


def test_real_train_loader_has_balanced_selected_counts() -> None:
    assert _dataset("train").class_counts == {
        "IDLE": 195,
        "WAITING": 294,
        "READY": 147,
        "NONE": 180,
    }


def test_real_validation_loader_uses_natural_distribution() -> None:
    assert _dataset("validation").class_counts == {
        "IDLE": 78,
        "WAITING": 357,
        "READY": 87,
        "NONE": 77,
    }


def test_real_test_loader_uses_natural_distribution() -> None:
    assert _dataset("test").class_counts == {
        "IDLE": 156,
        "WAITING": 598,
        "READY": 176,
        "NONE": 252,
    }
